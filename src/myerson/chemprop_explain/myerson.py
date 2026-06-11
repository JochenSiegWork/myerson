import logging
import numpy as np

try:
    import chemprop
except ImportError:
    raise ImportError("Failed to import chemprop. MPNN explanations not available.")
import torch
from tqdm import tqdm

from chemprop.models.model import MPNN
from chemprop.data.molgraph import MolGraph
from chemprop.data.collate import BatchMolGraph

from myerson import MyersonCalculator, MyersonSampler
from myerson.chemprop_explain.utils import to_networkx


class MyersonExplainer(MyersonCalculator):
    r"""Explains the prediction of a chemprop MPNN with Myerson values.
        The MPNN is treated as the coalition function of a game and its prediction
        as the payoff of the game. The Myerson values show how much each node of 
        the graph contributed to the final prediction.

    Args:
        molgraph (MolGraph): The chemprop MolGraph instance that is to be explained.
        coalition_function (MPNN): The message passing neural network.
        disable_tqdm (bool, optional): Disables progress bar. Defaults to True.
    """

    def __init__(self, 
                molgraph: MolGraph,
                coalition_function: MPNN,
                disable_tqdm: bool=True) -> None:
        """Instantiate the class.
        """

        self.disable_tqdm = disable_tqdm
        self.log = logging.getLogger("MyersonExplainer")

        self.molgraph = molgraph
        self.coalition_function = coalition_function

        self.nx_graph = to_networkx(molgraph)
        self.grand_coalition = list(self.nx_graph.nodes()) # alias: set of players / set of nodes / F
        cc = self.number_connected_components()
        if cc > 1:
            self.log.warning(f"Your graph has {cc} individual components. The worth"
                        " of the grand coalition and the prediction of a GNN can"
                        " differ.")
            pred = self.calculate_prediction()
            worth = self.calculate_worth_of_grand_coalition()
            self.log.warning(f"Prediction={pred:.4f}, Worth={worth:.4f}")

    def _forward(self, batch_mol_graph: BatchMolGraph) -> torch.Tensor:
        """Run the coalition function (MPNN) in eval mode without building an
        autograd graph. Returns a detached CPU tensor of shape ``(B, tasks)``.

        The model's original ``training`` flag is restored afterwards so calling
        an explainer does not silently mutate the user's model.
        """
        model = self.coalition_function
        was_training = model.training
        model.eval()
        try:
            batch_mol_graph.to(model.device)
            with torch.no_grad():
                return model(batch_mol_graph).detach().cpu()
        finally:
            model.train(was_training)

    def _empty_worth(self):
        """Worth assigned to the empty coalition."""
        return 0.0

    def _postprocess_worth(self, model_output_row: torch.Tensor) -> float:
        """Convert a single row of the (batched) model output into a worth."""
        return model_output_row.item()

    def calculate_worth_of_single_graph_restricted_coalition(self,
        graph_restricted_coalition: tuple,
        molgraph: MolGraph) -> float:
        """Calculate the worth of a graph restricted coalition, i. e. a single
        connected component.

        Args:
            graph_restricted_coalition (tuple): Graph restricted coalition as
                node indices.
            molgraph (MolGraph): Graph from which a subgraph
                of the connected components will be extracted according to the
                graph restricted coalition.

        Returns:
            float: Worth, the output of the coalition function for the connected
            subgraph. 
        """
        if graph_restricted_coalition == ():
            return self._empty_worth()
        subgraph = self.subgraph_from_coalition(graph_restricted_coalition, molgraph)
        out = self._forward(BatchMolGraph([subgraph]))
        return self._postprocess_worth(out.squeeze(0))

    def calculate_worth_of_graph_restricted_coalitions(self,
        graph_restricted_coalitions: list,
        batch_size: int = 1024) -> dict:
        """Calculate the worth of every graph restricted coalition and map it to
        its worth.

        The non-empty connected components are evaluated in *batches*: many
        subgraphs are collated into a single ``BatchMolGraph`` and run through
        the MPNN in one forward pass (under ``torch.no_grad`` + ``eval``). This
        is dramatically faster than one forward pass per coalition.

        Args:
            graph_restricted_coalitions (list): Set of connected components as
                tuples of node indices.
            batch_size (int, optional): Maximum number of subgraphs per forward
                pass. Limits peak memory for graphs with many components.
                Defaults to 1024.

        Returns:
            dict: Dictionary mapping each connected component to its worth.
        """
        self.log.info(f"Calculating worth of graph restricted coalitions.")
        graph_restricted_coalitions_to_worth = {}

        # Separate the empty coalition (no forward pass needed).
        non_empty = []
        for coalition in graph_restricted_coalitions:
            if coalition == ():
                graph_restricted_coalitions_to_worth[()] = self._empty_worth()
            else:
                non_empty.append(coalition)

        for start in tqdm(range(0, len(non_empty), batch_size),
                          desc="Calculating worth of graph restricted coalitions",
                          disable=self.disable_tqdm):
            chunk = non_empty[start:start + batch_size]
            out = self._forward(self._batch_mol_graph_from_coalitions(chunk))
            for i, coalition in enumerate(chunk):
                graph_restricted_coalitions_to_worth[coalition] = \
                    self._postprocess_worth(out[i])
        return graph_restricted_coalitions_to_worth

    def _batch_mol_graph_from_coalitions(self, coalitions: list) -> BatchMolGraph:
        """Build a :class:`BatchMolGraph` for many connected subgraphs in a
        single pass directly from the parent ``MolGraph``'s tensors.

        Equivalent to ``BatchMolGraph([subgraph_from_coalition(c) ...])`` but
        avoids creating one intermediate ``MolGraph`` object per coalition and
        avoids the second concatenation pass in ``BatchMolGraph.__post_init__``:
        the batched node/edge offsets are accumulated while masking, then
        concatenated once.

        TODO: speed up of this optimization is only 1.06x from caffeine to 1.21 naphthalene.
              Might be better to stick to chemprop API completely
        """
        mg = self.molgraph
        V_all, E_all = mg.V, mg.E
        edge_index, rev_edge_index = mg.edge_index, mg.rev_edge_index
        n_atoms = V_all.shape[0]
        n_edges = edge_index.shape[1]
        src, dst = edge_index[0], edge_index[1]

        Vs, Es, edge_indexes, rev_edge_indexes, batches = [], [], [], [], []
        node_offset = 0
        edge_offset = 0
        for i, coalition in enumerate(coalitions):
            nodes = np.sort(np.asarray(coalition, dtype=np.int64))
            k = nodes.shape[0]

            node_mask = np.zeros(n_atoms, dtype=bool)
            node_mask[nodes] = True
            Vs.append(V_all[nodes])

            edge_mask = node_mask[src] & node_mask[dst]
            n_sub_edges = int(edge_mask.sum())

            # Relabel kept nodes to local 0..k-1, then shift into the batch.
            node_relabel = np.empty(n_atoms, dtype=np.int64)
            node_relabel[nodes] = np.arange(k, dtype=np.int64)
            edge_indexes.append(node_relabel[edge_index[:, edge_mask]] + node_offset)
            Es.append(E_all[edge_mask])

            # Relabel kept edges to local 0..n_sub_edges-1, then shift.
            edge_relabel = np.full(n_edges, -1, dtype=np.int64)
            edge_relabel[edge_mask] = np.arange(n_sub_edges, dtype=np.int64)
            rev_edge_indexes.append(edge_relabel[rev_edge_index[edge_mask]] + edge_offset)

            batches.append(np.full(k, i, dtype=np.int64))
            node_offset += k
            edge_offset += n_sub_edges

        bmg = object.__new__(BatchMolGraph)
        bmg.V = torch.from_numpy(np.concatenate(Vs)).float()
        bmg.E = torch.from_numpy(
            np.concatenate(Es) if Es else np.empty((0, E_all.shape[1]), E_all.dtype)
        ).float()
        bmg.edge_index = torch.from_numpy(np.concatenate(edge_indexes, axis=1)).long()
        bmg.rev_edge_index = torch.from_numpy(np.concatenate(rev_edge_indexes)).long()
        bmg.batch = torch.from_numpy(np.concatenate(batches)).long()
        # name-mangled private size field on the slotted dataclass
        setattr(bmg, "_BatchMolGraph__size", len(coalitions))
        return bmg

    def calculate_worth_of_grand_coalition(self) -> float:
        """Calculate payoff of the game, i.e. the model prediction. Note that a
        disconnected graph (> 2 molecules) can lead to differeces between 
        the model prediction and this function. 

        Args:
            coalition_function (Callable): The coalition function associating a
                coalition with a payoff.
            nx_graph (nx.classes.graph.Graph): Coalition structure of the game
                as a graph.

        Returns:
            float: Payoff of the game / worth of grand coalition.
        """
        restricted_grand_coalition = self.restrict(self.grand_coalition, self.nx_graph)
        worth = sum([self.calculate_worth_of_single_graph_restricted_coalition(S, self.molgraph) \
                    for S in restricted_grand_coalition])
        return worth 

    def calculate_prediction(self) -> float:
        """Calculate the prediction of the GNN for the investigated graph. When 
        the graph is disconnected this prediction may differ from the worth 
        of the grand coalition.

        Returns:
            float: Prediction.
        """
        return self._forward(BatchMolGraph([self.molgraph])).item()

    def subgraph_from_coalition(self, graph_restricted_coalition: tuple, 
                                molgraph: MolGraph) -> MolGraph:
        """Generates a subgraph from a graph restricted coalition (a subset of
        nodes / players) and a graph.

        Args:
            nodes (tuple): Nodes which form the subgraph.
            molgraph (MolGraph): Subgraph induced in this graph by the subset of
                nodes.

        Returns:
            MolGraph: The new subgraph.
        """
        # unsorted nodes can result in the wrong edges
        nodes = sorted(graph_restricted_coalition)
        nodes = np.array(nodes, dtype=np.int32)
        node_mask = np.zeros(molgraph.V.shape[0], dtype=bool)
        node_mask[nodes] = True
        V = molgraph.V[node_mask]

        edge_mask = node_mask[molgraph.edge_index[0]] & node_mask[molgraph.edge_index[1]]
        edge_index = molgraph.edge_index[:, edge_mask]
        # fancy indexing to relabel edge_index, rev_edge_index
        node_idx = np.zeros(node_mask.size, dtype=np.int32)
        node_idx[nodes] = np.arange(node_mask.sum())
        edge_index = node_idx[edge_index]

        E = molgraph.E[edge_mask]

        edge_idx_map = np.full(molgraph.edge_index.shape[1], -1, dtype=np.int32)
        edge_idx_map[edge_mask] = np.arange(edge_mask.sum())
        edge_idx_map
        rev_edge_index_masked = molgraph.rev_edge_index[edge_mask]
        rev_edge_index = edge_idx_map[rev_edge_index_masked]

        subgraph = MolGraph(V=V, E=E, edge_index=edge_index, rev_edge_index=rev_edge_index)
        return subgraph

class MyersonSamplingExplainer(MyersonSampler, MyersonExplainer):
    """A class explaining GNN predictions with approximated Myerson values.

    Args:
        molgraph (MolGraph): The chemprop MolGraph instance that is to be explained.
        coalition_function (MPNN): The message passing neural network.
        seed (None | int, optional): Seed for randomness. Defaults to None.
        number_of_samples (int, optional): Number of sampling steps. Defaults to 1000.
        disable_tqdm (bool, optional): Disables progress bar. Defaults to True.
    """
    def __init__(self,
                molgraph: MolGraph,
                coalition_function: MPNN,
                seed: None | int = None, 
                number_of_samples: int = 1000,
                disable_tqdm: bool=True) -> None:
        """Instantiates the class.
        """
        self.disable_tqdm = disable_tqdm
        self.log = logging.getLogger("MyersonSamplingExplainer")

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.number_of_samples = number_of_samples

        self.molgraph = molgraph
        self.coalition_function = coalition_function

        self.nx_graph = to_networkx(molgraph)
        self.grand_coalition = list(self.nx_graph.nodes()) # alias: set of players / set of nodes / F
        cc = self.number_connected_components()
        if cc > 1:
            self.log.warning(f"Your graph has {cc} individual components. The worth"
                        " of the grand coalition and the prediction of a GNN can"
                        " differ.")
            pred = self.calculate_prediction()
            worth = self.calculate_worth_of_grand_coalition()
            self.log.warning(f"Prediction={pred:.4f}, Worth={worth:.4f}")


class MyersonClassExplainer(MyersonExplainer):
    r"""Explains the prediction of a graph neural network (GNN) classifier with
        Myerson values.  The GNN is treated as the coalition function of a game
        and its prediction as the payoff of the game. The Myerson values show
        how much each node of the graph contributed to the final prediction.

    Args:
        molgraph (MolGraph): The chemprop MolGraph instance that is to be explained.
        coalition_function (MPNN): The message passing neural network.
        disable_tqdm (bool, optional): Disables progress bar. Defaults to True.
    """

    # Multi-output worths are tensors; use the generic (tensor-capable) path.
    _supports_subset_dp = False

    def __init__(self,
                molgraph: MolGraph,
                coalition_function: MPNN,
                disable_tqdm: bool=True) -> None:
        """Instantiate the class.
        """

        self.disable_tqdm = disable_tqdm
        self.log = logging.getLogger("MyersonClassExplainer")
        self.molgraph = molgraph
        self.coalition_function = coalition_function

        self.nx_graph = to_networkx(molgraph)
        self.grand_coalition = list(self.nx_graph.nodes()) # alias: set of players / set of nodes / F
        self.pred = self.calculate_prediction()
        cc = self.number_connected_components()
        if cc > 1:
            self.log.warning(f"Your graph has {cc} individual components. The worth"
                        " of the grand coalition and the prediction of a GNN can"
                        " differ.")
            pred = self.calculate_prediction()
            worth = self.calculate_worth_of_grand_coalition()
            self.log.warning(f"Prediction={pred}, Worth={worth}")
    
    def _empty_worth(self) -> torch.Tensor:
        """Worth assigned to the empty coalition (zero vector over tasks)."""
        return torch.zeros(self.pred.shape)

    def _postprocess_worth(self, model_output_row: torch.Tensor) -> torch.Tensor:
        """Keep the full per-task output vector for the (multi-output) classifier."""
        return model_output_row.clone()

    def calculate_prediction(self) -> torch.Tensor:
        """Calculate the prediction of the GNN for the investigated graph. When 
        the graph is disconnected this prediction may differ from the worth 
        of the grand coalition.

        Returns:
            torch.Tensor: Prediction.
        """
        return self._forward(BatchMolGraph([self.molgraph])).squeeze(0)


class MyersonSamplingClassExplainer(MyersonSamplingExplainer, MyersonClassExplainer):
    """A class explaining a GNNs classifier predictions with approximated Myerson values.

    Args:
        molgraph (MolGraph): The chemprop MolGraph instance that is to be explained.
        coalition_function (MPNN): The message passing neural network.
        seed (None | int, optional): Seed for randomness. Defaults to None.
        number_of_samples (int, optional): Number of sampling steps. Defaults to 1000.
        disable_tqdm (bool, optional): Disables progress bar. Defaults to True.
    """
    def __init__(self,
                molgraph: MolGraph,
                coalition_function: MPNN,
                seed: None | int = None, 
                number_of_samples: int = 1000,
                disable_tqdm: bool=True) -> None:
        """Instantiates the class.
        """
        self.disable_tqdm = disable_tqdm
        self.log = logging.getLogger("MyersonSamplingClassExplainer")

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.number_of_samples = number_of_samples

        self.molgraph = molgraph
        self.coalition_function = coalition_function

        self.nx_graph = to_networkx(molgraph)
        self.grand_coalition = list(self.nx_graph.nodes()) # alias: set of players / set of nodes / F
        self.pred = self.calculate_prediction()
        cc = self.number_connected_components()
        if cc > 1:
            self.log.warning(f"Your graph has {cc} individual components. The worth"
                        " of the grand coalition and the prediction of a GNN can"
                        " differ.")
            pred = self.calculate_prediction()
            worth = self.calculate_worth_of_grand_coalition()
            self.log.warning(f"Prediction={pred}, Worth={worth}")

    def map_coalition_to_worth(self, coalitions: list[tuple], 
                       coalitions_to_graph_restricted_coalitions: dict,
                       graph_restricted_coalitions_to_worth: dict) -> dict:
        """Map every coalition to its worth.

        Args:
            coalitions (list): List of all coalitions (2^{num_nodes}). 
            coalitions_to_graph_restricted_coalitions (dict): Dictionary mapping
                the coalitions to the corresponding graph restricted coalitions.
            graph_restricted_coalitions_to_worth (dict): Dictionary mapping the 
                graph restricted coalitions to their worth.

        Returns:
            dict: Dictionary mapping each coalition to its worth.
        """
        self.log.info(f"Mapping coalitions to worth.")
        coalition_to_worth = {}
        for coalition in tqdm(coalitions, desc="Mapping coalitions to worth", disable=self.disable_tqdm):
            worth = torch.zeros(self.pred.shape)
            for graph_restricted_coalition in coalitions_to_graph_restricted_coalitions[coalition]:
                worth += graph_restricted_coalitions_to_worth[graph_restricted_coalition]
            coalition_to_worth.update({coalition: worth})
        return coalition_to_worth
    
    def sample_all_myerson_values(self) -> np.ndarray:
        """Use Monte Carlo sampling to approximate the Myerson values for every
        node / player in the graph.

        Returns:
            np.ndarray: Sampled Myerson values.
        """
        self.sample_all_mappings()
        self.log.info(f"Calculating sampled Myerson values.")
        # Per-task worth vectors keyed by integer bitmask (see base class).
        worth = {mask: np.asarray(t).squeeze() for mask, t in self._worth_by_mask.items()}
        node_bits = self._node_bits
        random_node_bit = self._random_node_bit
        n = len(node_bits)
        n_tasks = self.pred.shape[0]

        my_values = np.zeros((n, n_tasks), dtype=float)
        for mask in tqdm(self._base_masks, disable=self.disable_tqdm,
                         desc="Calculate sampled Myerson values"):
            for j in range(n):
                bit = node_bits[j]
                without = ((mask ^ bit) | random_node_bit) if (mask & bit) else mask
                my_values[j] += worth[without | bit] - worth[without]

        my_values = my_values / self.number_of_samples
        log_string = "".join([f"\t{node}: {val}\n" for node, val in zip(self.grand_coalition, my_values)])
        self.log.info(f"Sampled Myerson Values:\n{log_string}")
        return my_values

def explain(molgraph: MolGraph,
            model: MPNN,
            sample_if_more_nodes_than: int=20,
            verbose: bool=False) -> dict:
    """A function to quickly get started with explaining GNN predictions using Myerson values.

    Args:
        molgraph (MolGraph): The chemprop MolGraph instance.
        model (MPNN): The message passing neural network.
        sample_if_more_nodes_than (int, optional): Barrier for when to start
            sampling instead of exact calculations. Defaults to 20.
        verbose (bool, optional): Whether to log information to the output and
            show progress bars. Defaults to False.

    Returns:
        dict: The (sampled) Myerson values.
    """

    if verbose:
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] %(message)s', force=True)
        disable_tqdm=False
    else:
        disable_tqdm=True

    node_count = molgraph.V.shape[0]
    if node_count > sample_if_more_nodes_than:
        logging.info("Sampling Myerson values.")
        sampler = MyersonSamplingExplainer(molgraph, model, disable_tqdm=disable_tqdm)
        return sampler.sample_all_myerson_values()
    else:
        logging.info("Calculating exact Myerson values.")
        explainer = MyersonExplainer(molgraph, model, disable_tqdm=disable_tqdm)
        return explainer.calculate_all_myerson_values()
