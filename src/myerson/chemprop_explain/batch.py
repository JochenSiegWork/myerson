"""Cross-molecule batched Myerson explanations for chemprop MPNNs.

This module holds the *batch* layer on top of the per-molecule explainers in
:mod:`myerson.chemprop_explain.myerson`:

* :func:`explain_batch` — pools the (expensive) NN forward passes of many
  molecules' sampled subgraphs into shared batched forwards, so the GPU stays
  saturated instead of being under-filled one molecule at a time.
* :class:`ChempropBatchExplainer` — the high-level, model-resident front door
  that accepts a single ``MolGraph``, a list of them, or a ``BatchMolGraph``,
  streams the input in memory-bounded groups, and returns one Myerson-value
  array per molecule.

Both support regression (scalar worth) and classification (per-task tensor
worth) models via the ``classification`` flag, which routes to the
``*ClassExplainer`` variants.
"""

import logging
import math

import numpy as np
import torch
from tqdm import tqdm

from chemprop.models.model import MPNN
from chemprop.data.molgraph import MolGraph
from chemprop.data.collate import BatchMolGraph

from myerson.chemprop_explain.utils import unbatch
from myerson.chemprop_explain.myerson import (
    MyersonExplainer,
    MyersonSamplingExplainer,
    MyersonClassExplainer,
    MyersonSamplingClassExplainer,
)


def explain_batch(molgraphs: list,
                  model: MPNN,
                  seed: None | int = None,
                  number_of_samples: int = 1000,
                  sample_if_more_nodes_than: int = 20,
                  max_atoms_per_forward: int = 300_000,
                  classification: bool = False,
                  verbose: bool = False) -> list:
    """Explain a *batch* of molecules, pooling the NN forward passes across
    molecules so the GPU stays saturated instead of being under-filled one
    molecule at a time.

    Molecules with ``> sample_if_more_nodes_than`` nodes are explained by
    sampling, and **their connected subgraphs are pooled into shared batched
    forward passes** (the expensive part). Smaller molecules are computed exactly
    per-molecule (already cheap). Results are identical to explaining each
    molecule individually: for regression that is
    :func:`~myerson.chemprop_explain.myerson.explain`; for
    ``classification=True`` it is the per-molecule ``MyersonClassExplainer`` /
    ``MyersonSamplingClassExplainer`` (bit-identical for a fixed ``seed`` on the
    sampled molecules).

    GPU memory is bounded by ``max_atoms_per_forward`` (each forward carries at
    most that many subgraph atoms, splitting even a single large molecule's
    components across several forwards). *Host* memory grows with the number of
    molecules passed in (their sampled bookkeeping is held at once); for large
    datasets, drive this through :class:`ChempropBatchExplainer`, which streams
    the input in memory-bounded groups.

    Args:
        molgraphs (list[MolGraph]): The molecules to explain.
        model (MPNN): The message passing neural network.
        seed (int | None, optional): Seed for the sampled molecules. Defaults to None.
        number_of_samples (int, optional): Sampling steps. Defaults to 1000.
        sample_if_more_nodes_than (int, optional): Exact/sampling switch per
            molecule. Defaults to 20.
        max_atoms_per_forward (int, optional): Max total subgraph atoms per pooled
            forward pass; bounds GPU memory regardless of molecule size. Defaults
            to 300_000.
        classification (bool, optional): If ``True``, explain a multi-output
            (classifier) model: routes to ``MyersonClassExplainer`` /
            ``MyersonSamplingClassExplainer`` and returns per-molecule arrays of
            shape ``(n_atoms, n_tasks)`` of per-task tensor worths. If ``False``
            (default), the model is treated as single-output regression and each
            molecule yields a ``(n_atoms,)`` array.
        verbose (bool, optional): INFO logging + progress bars. Defaults to False.

    Returns:
        list: One Myerson-value array per input molecule (same order).
    """
    if verbose:
        logging.basicConfig(level=logging.INFO,
                            format='[%(asctime)s - %(levelname)s] %(message)s', force=True)
        disable_tqdm = False
    else:
        disable_tqdm = True

    exact_cls = MyersonClassExplainer if classification else MyersonExplainer
    sampling_cls = (MyersonSamplingClassExplainer if classification
                    else MyersonSamplingExplainer)

    results: list = [None] * len(molgraphs)
    exact = []     # (original_index, explainer, sub_masks, sub_tuples, neighbour_masks)
    sampling = []  # (original_index, sampler)

    # Phase 0: route molecules. Exact molecules are enumerated now but their
    # connected-subgraph worths are pooled below, just like sampled molecules, so
    # datasets of many small/medium molecules do not under-fill the GPU.
    for i, mg in enumerate(molgraphs):
        if mg.V.shape[0] > sample_if_more_nodes_than:
            sampler = sampling_cls(
                mg, model, seed=seed, number_of_samples=number_of_samples,
                disable_tqdm=disable_tqdm)
            # TODO use of private function. Needs Interface change?
            sampler.random_node, sampler.permutations_without_random_node = \
                sampler._sample_base_permutations(number_of_samples)
            sampling.append((i, sampler))
        else:
            explainer = exact_cls(mg, model, disable_tqdm=disable_tqdm)
            exact.append((i, explainer,
                          *explainer._enumerate_connected_subgraphs()))

    # Phase E: pooled exact connected-subgraph worths + exact value reduction.
    if exact:
        exact_worth_pooled = _pooled_worths_by_parent(
            [(item[1], item[3]) for item in exact], host=exact[0][1],
            max_atoms_per_forward=max_atoms_per_forward,
            disable_tqdm=disable_tqdm,
            desc="Pooled exact worth (atoms)")
        for k, (i, explainer, sub_masks, sub_tuples, neighbour_masks) in enumerate(exact):
            worth_by_tuple = {c: exact_worth_pooled[(k, c)] for c in sub_tuples}
            explainer.graph_restricted_coalitions = set(sub_tuples)
            explainer.graph_restricted_coalitions_to_worth = worth_by_tuple
            results[i] = _finish_connected_enum_values_from_worths(
                explainer, sub_masks, sub_tuples, neighbour_masks, worth_by_tuple)
        exact_worth_pooled.clear()

    if not sampling:
        return results

    # Phase A: enumerate each molecule's connected components (no NN yet).
    components_per = []
    for k, (_, sampler) in enumerate(sampling):
        components_per.append(sampler._enumerate_sampling_components())

    # Phase B: pooled NN forward, chunked by a *total-atom* budget so GPU memory
    # stays bounded regardless of molecule size — even a single huge molecule's
    # components are split across several forwards.
    worth_pooled = _pooled_worths_by_parent(
        [(sampler, components_per[k]) for k, (_, sampler) in enumerate(sampling)],
        host=sampling[0][1], max_atoms_per_forward=max_atoms_per_forward,
        disable_tqdm=disable_tqdm,
        desc="Pooled sampled worth (atoms)")

    # Phase C: scatter worths back, finish each molecule's value loop, and free
    # its (potentially large) bookkeeping immediately to bound host memory.
    for k, (i, sampler) in enumerate(sampling):
        grc_to_worth = {}
        for c in components_per[k]:
            # TODO sampler private function
            grc_to_worth[c] = (sampler._empty_worth() if c == ()
                               else worth_pooled[(k, c)])
        sampler._finish_sampling_worth_by_mask(grc_to_worth)
        results[i] = sampler._sampled_myerson_values_from_prepared()
        # Release this molecule's worths from the shared pool as we go.
        for c in components_per[k]:
            worth_pooled.pop((k, c), None)
        components_per[k] = None

    return results


def _pooled_worths_by_parent(parent_components: list,
                             host,
                             max_atoms_per_forward: int,
                             disable_tqdm: bool,
                             desc: str = "Pooled batched worth (atoms)") -> dict:
    """Evaluate connected-component worths pooled across parent molecules.

    Args:
        parent_components: list of ``(explainer_or_sampler, components)``. The
            parent index in this list is used as the first key in the returned
            dict.
        host: Any compatible explainer/sampler used to run the merged forward.
        max_atoms_per_forward: Atom budget for each pooled forward. A single
            component larger than the budget is allowed to form a one-component
            oversized chunk; otherwise chunks are flushed *before* adding a
            component that would exceed the budget.
        disable_tqdm: Disable progress bar.
        desc: Progress-bar description.

    Returns:
        dict: ``(parent_index, component_tuple) -> worth`` for every non-empty
        component. Empty components never require a model forward and are handled
        by the caller via ``_empty_worth``.
    """
    total_atoms = sum(len(c) for _, comps in parent_components for c in comps)
    worth_pooled: dict = {}

    def _flush(pending_by_k):
        if not pending_by_k:
            return
        bmgs, order = [], []
        for k, comps in pending_by_k.items():
            parent = parent_components[k][0]
            bmgs.append(parent._batch_mol_graph_from_coalitions(comps))
            order.append((k, parent, comps))
        out = host._forward(_merge_batch_mol_graphs(bmgs))
        offset = 0
        for (k, parent, comps) in order:
            for j, c in enumerate(comps):
                worth_pooled[(k, c)] = parent._postprocess_worth(out[offset + j])
            offset += len(comps)

    pbar = tqdm(total=total_atoms, desc=desc, disable=disable_tqdm, unit="atom")
    pending_by_k: dict = {}
    pending_atoms = 0
    for k, (_, comps) in enumerate(parent_components):
        for c in comps:
            n_atoms = len(c)
            if n_atoms == 0:
                continue
            # Flush BEFORE adding the component to keep chunks within the atom
            # budget whenever possible. If a single component is larger than the
            # budget, it is forwarded alone (pending_atoms == 0 case).
            if pending_atoms and pending_atoms + n_atoms > max_atoms_per_forward:
                _flush(pending_by_k)
                pbar.update(pending_atoms)
                pending_by_k, pending_atoms = {}, 0
            pending_by_k.setdefault(k, []).append(c)
            pending_atoms += n_atoms
            if pending_atoms >= max_atoms_per_forward:
                _flush(pending_by_k)
                pbar.update(pending_atoms)
                pending_by_k, pending_atoms = {}, 0
    if pending_by_k:
        _flush(pending_by_k)
        pbar.update(pending_atoms)
    pbar.close()
    return worth_pooled


def _finish_connected_enum_values_from_worths(explainer,
                                              sub_masks: list,
                                              sub_tuples: list,
                                              neighbour_masks: list,
                                              worth_by_tuple: dict) -> np.ndarray:
    """Finish Skibski connected-subgraph exact Myerson values from pooled worths.

    This is the same reduction as
    ``MyersonCalculator._calculate_all_myerson_values_connected_enum`` after the
    worth oracle has been evaluated, factored locally so ``explain_batch`` can
    pool exact-path Chemprop forwards across molecules.
    """
    _, label_to_index, _ = explainer._get_adjacency_masks(explainer.nx_graph)
    n = len(label_to_index)
    if n == 0:
        return np.array([], dtype=np.float64)

    sample = next(iter(worth_by_tuple.values())) if worth_by_tuple else 0.0
    value0, scalar = explainer._coerce_worth(sample)
    if scalar:
        mv = np.zeros(n, dtype=np.float64)
    else:
        mv = np.zeros((n, value0.shape[0]), dtype=np.float64)

    fact = [math.factorial(k) for k in range(n + 1)]
    for mask, tup, nbr in zip(sub_masks, sub_tuples, neighbour_masks):
        value, _ = explainer._coerce_worth(worth_by_tuple[tup])
        if scalar and value == 0.0:
            continue
        s = len(tup)
        t = nbr.bit_count()
        denom = fact[s + t]
        pos = fact[s - 1] * fact[t] / denom
        m = mask
        while m:
            low = m & (-m)
            mv[low.bit_length() - 1] += pos * value
            m ^= low
        if t:
            neg = fact[s] * fact[t - 1] / denom
            c = nbr
            while c:
                low = c & (-c)
                mv[low.bit_length() - 1] -= neg * value
                c ^= low

    order = [label_to_index[label] for label in explainer.grand_coalition]
    return mv[order]


def _merge_batch_mol_graphs(bmgs: list) -> BatchMolGraph:
    """Merge several ``BatchMolGraph`` objects into one (disjoint union), so the
    subgraphs of *many molecules* are run through the MPNN in a single forward.

    Concatenates the node/edge feature tensors and offsets ``edge_index`` (by
    node count), ``rev_edge_index`` (by edge count) and ``batch`` (by graph
    count). Each input is itself a batch of subgraphs from one parent molecule
    (built by the vectorized §2.7 builder).
    """
    if len(bmgs) == 1:
        return bmgs[0]

    Vs, Es, edge_indices, revs, batches = [], [], [], [], []
    node_off = edge_off = graph_off = 0
    for b in bmgs:
        Vs.append(b.V)
        Es.append(b.E)
        edge_indices.append(b.edge_index + node_off)
        revs.append(b.rev_edge_index + edge_off)
        batches.append(b.batch + graph_off)
        node_off += b.V.shape[0]
        edge_off += b.E.shape[0]
        graph_off += int(getattr(b, "_BatchMolGraph__size"))

    merged = object.__new__(BatchMolGraph)
    merged.V = torch.cat(Vs, dim=0)
    merged.E = torch.cat(Es, dim=0)
    merged.edge_index = torch.cat(edge_indices, dim=1)
    merged.rev_edge_index = torch.cat(revs, dim=0)
    merged.batch = torch.cat(batches, dim=0)
    setattr(merged, "_BatchMolGraph__size", graph_off)
    return merged


class ChempropBatchExplainer:
    """High-level Myerson explainer for one molecule, many molecules, or a
    ``BatchMolGraph``, the recommended entry point for explaining datasets.

    Construct it once with a model (held resident, on its device) and reuse it
    across the whole dataset. Each call routes every molecule to the exact or
    sampling explainer by node count, and **pools the sampling-path NN forwards
    across molecules** (see :func:`explain_batch`) to keep the GPU saturated.

    Input/output cardinality:
        * a single ``MolGraph``                  -> a single ``np.ndarray``
        * an iterable of ``MolGraph``            -> ``list[np.ndarray]``
        * a ``BatchMolGraph`` (collated molecules) -> ``list[np.ndarray]``

    Each per-molecule array is ``(n_atoms,)`` for regression and
    ``(n_atoms, n_tasks)`` for ``classification=True``. Results are identical to
    explaining each molecule individually: for regression that is
    :func:`~myerson.chemprop_explain.myerson.explain`; for classification it is
    the per-molecule ``MyersonClassExplainer`` /
    ``MyersonSamplingClassExplainer``. Either way the sampled molecules are
    bit-identical for a fixed ``seed``.

    Args:
        model (MPNN): The message passing neural network (used as-is, on its
            current device). Use :meth:`from_checkpoint` to load + place it once.
        sample_if_more_nodes_than (int, optional): Per-molecule exact/sampling
            switch. Defaults to 18.
        number_of_samples (int, optional): Sampling steps. Defaults to 1000.
        seed (int | None, optional): Seed for the sampled molecules. Defaults to None.
        max_atoms_per_group (int, optional): Molecules are streamed in groups
            whose summed atom count stays under this budget — this bounds *host*
            RAM (the sampled bookkeeping grows ~``samples × atoms`` per molecule),
            so a few huge molecules form tiny groups while many small ones pool
            into large groups. Defaults to 1500.
        max_atoms_per_forward (int, optional): Max total subgraph atoms per NN
            forward; bounds *GPU* memory regardless of molecule size. Defaults to
            300_000.
        group_size (int, optional): Hard cap on molecules per group (safety net on
            top of the atom budget). Defaults to 512.
        classification (bool, optional): Explain a multi-output (classifier)
            model; each molecule yields a ``(n_atoms, n_tasks)`` array. Defaults
            to ``False`` (single-output regression).
        verbose (bool, optional): INFO logging + progress bars. Defaults to False.
    """

    def __init__(self,
                 model: MPNN,
                 sample_if_more_nodes_than: int = 18,
                 number_of_samples: int = 1000,
                 seed: None | int = None,
                 max_atoms_per_group: int = 1500,
                 max_atoms_per_forward: int = 300_000,
                 group_size: int = 512,
                 classification: bool = False,
                 verbose: bool = False) -> None:
        self.model = model
        self.sample_if_more_nodes_than = sample_if_more_nodes_than
        self.number_of_samples = number_of_samples
        self.seed = seed
        self.max_atoms_per_group = max_atoms_per_group
        self.max_atoms_per_forward = max_atoms_per_forward
        self.group_size = group_size
        self.classification = classification
        self.verbose = verbose

    @classmethod
    def from_checkpoint(cls, checkpoint, map_location: str | None = None,
                        **kwargs) -> "ChempropBatchExplainer":
        """Load an MPNN checkpoint (once), place it on a device, and wrap it.

        Args:
            checkpoint: Path to the chemprop MPNN checkpoint.
            map_location (str | None): Device; defaults to CUDA if available.
            **kwargs: Forwarded to :class:`ChempropBatchExplainer`.
        """
        device = map_location or ("cuda:0" if torch.cuda.is_available() else "cpu")
        model = MPNN.load_from_checkpoint(str(checkpoint), map_location=device)
        model.eval()
        return cls(model, **kwargs)

    def _normalize(self, inputs) -> tuple[bool, list]:
        """Return ``(is_single, list_of_molgraphs)`` for any accepted input."""
        if isinstance(inputs, BatchMolGraph):
            return False, unbatch(inputs)
        if isinstance(inputs, MolGraph):
            return True, [inputs]
        return False, list(inputs)

    def explain(self, inputs):
        """Explain a molecule / molecules / ``BatchMolGraph``.

        Returns a single ``np.ndarray`` for a single ``MolGraph`` input, else a
        ``list[np.ndarray]`` (one per molecule, input order preserved).
        """
        is_single, molgraphs = self._normalize(inputs)
        results = list(self._iter_groups(molgraphs))
        return results[0] if is_single else results

    def explain_iter(self, inputs):
        """Stream explanations one molecule at a time (input order preserved).

        Processes the input in memory-bounded groups (``max_atoms_per_group``)
        so peak host memory stays bounded for very large datasets; yields each
        molecule's values as soon as its group finishes. Always yields per
        molecule (even for a single input).
        """
        _, molgraphs = self._normalize(inputs)
        yield from self._iter_groups(molgraphs)

    def _grouped(self, molgraphs: list):
        """Yield consecutive molecule groups bounded by the atom budget (and the
        ``group_size`` hard cap). A single molecule larger than the budget forms
        its own group."""
        group, atoms = [], 0
        for mg in molgraphs:
            n = mg.V.shape[0]
            if group and (atoms + n > self.max_atoms_per_group
                          or len(group) >= self.group_size):
                yield group
                group, atoms = [], 0
            group.append(mg)
            atoms += n
        if group:
            yield group

    def _iter_groups(self, molgraphs: list):
        """Run ``explain_batch`` group-by-group, yielding per-molecule results."""
        for group in self._grouped(molgraphs):
            yield from explain_batch(
                group, self.model, seed=self.seed,
                number_of_samples=self.number_of_samples,
                sample_if_more_nodes_than=self.sample_if_more_nodes_than,
                max_atoms_per_forward=self.max_atoms_per_forward,
                classification=self.classification,
                verbose=self.verbose)

