"""`ChempropBatchExplainer`: single / list / BatchMolGraph input parity.

The high-level explainer must return exactly what the lower-level `explain` /
`explain_batch` return, for every accepted input shape, and `explain_iter` must
stream the same values — for both regression and classification models.
"""

import numpy as np
import pytest
import torch
from chemprop.models.model import MPNN
from chemprop.data import MoleculeDatapoint, MoleculeDataset
from chemprop.data.collate import BatchMolGraph

from myerson.chemprop_explain import (
    MyersonExplainer,
    MyersonSamplingExplainer,
    MyersonClassExplainer,
    MyersonSamplingClassExplainer,
    ChempropBatchExplainer,
)

CKPT = "tests/chemprop_regression.ckpt"
CLASS_CKPT = "tests/chemprop_multitask.ckpt"
SMILES = [
    "CCO",                        # ethanol  (N=3, exact)
    "c1ccccc1O",                  # phenol   (N=7)
    "CC(=O)Oc1ccccc1C(=O)O",      # aspirin  (N=13)
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O", # caffeine (N=14)
]
SEED = 42
N_SAMPLES = 150
THRESHOLD = 5  # ethanol exact; rest sampled


@pytest.fixture(scope="module")
def model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    m = MPNN.load_from_checkpoint(CKPT, map_location=device)
    m.eval()
    return m


@pytest.fixture(scope="module")
def class_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    m = MPNN.load_from_checkpoint(CLASS_CKPT, map_location=device)
    m.eval()
    return m


def _mgs():
    return [MoleculeDataset([MoleculeDatapoint.from_smi(s)])[0].mg for s in SMILES]


def _reference(mg, model):
    if mg.V.shape[0] > THRESHOLD:
        return MyersonSamplingExplainer(
            mg, model, seed=SEED, number_of_samples=N_SAMPLES
        ).sample_all_myerson_values()
    return MyersonExplainer(mg, model).calculate_all_myerson_values()


def _reference_class(mg, model):
    if mg.V.shape[0] > THRESHOLD:
        return MyersonSamplingClassExplainer(
            mg, model, seed=SEED, number_of_samples=N_SAMPLES
        ).sample_all_myerson_values()
    return MyersonClassExplainer(mg, model).calculate_all_myerson_values()


@pytest.fixture(scope="module")
def explainer(model):
    return ChempropBatchExplainer(
        model, sample_if_more_nodes_than=THRESHOLD,
        number_of_samples=N_SAMPLES, seed=SEED)


class TestChempropBatchExplainer:

    def test_single_molgraph_returns_array(self, explainer, model):
        mg = _mgs()[1]  # phenol
        out = explainer.explain(mg)
        assert isinstance(out, np.ndarray)
        np.testing.assert_allclose(out, _reference(mg, model), atol=1e-6)

    def test_list_of_molgraphs(self, explainer, model):
        mgs = _mgs()
        out = explainer.explain(mgs)
        assert isinstance(out, list) and len(out) == len(mgs)
        for o, mg in zip(out, mgs):
            np.testing.assert_allclose(o, _reference(mg, model), atol=1e-6)

    def test_batch_mol_graph_input(self, explainer, model):
        mgs = _mgs()
        out = explainer.explain(BatchMolGraph(list(mgs)))
        assert isinstance(out, list) and len(out) == len(mgs)
        for o, mg in zip(out, mgs):
            np.testing.assert_allclose(o, _reference(mg, model), atol=1e-6)

    def test_explain_iter_streams_same_values(self, explainer, model):
        mgs = _mgs()
        streamed = list(explainer.explain_iter(mgs))
        assert len(streamed) == len(mgs)
        for o, mg in zip(streamed, mgs):
            np.testing.assert_allclose(o, _reference(mg, model), atol=1e-6)

    def test_grouping_is_transparent(self, model):
        """A tiny group_size must give the same results as one big group."""
        mgs = _mgs()
        big = ChempropBatchExplainer(model, sample_if_more_nodes_than=THRESHOLD,
                                     number_of_samples=N_SAMPLES, seed=SEED,
                                     group_size=100).explain(mgs)
        small = ChempropBatchExplainer(model, sample_if_more_nodes_than=THRESHOLD,
                                       number_of_samples=N_SAMPLES, seed=SEED,
                                       group_size=1).explain(mgs)
        for a, b in zip(big, small):
            np.testing.assert_allclose(a, b, atol=1e-6)

    def test_classification_list(self, class_model):
        """classification=True must match the per-molecule class explainers and
        return (n_atoms, n_tasks) arrays."""
        mgs = _mgs()
        out = ChempropBatchExplainer(
            class_model, sample_if_more_nodes_than=THRESHOLD,
            number_of_samples=N_SAMPLES, seed=SEED,
            classification=True).explain(mgs)
        assert isinstance(out, list) and len(out) == len(mgs)
        for o, mg in zip(out, mgs):
            ref = np.asarray(_reference_class(mg, class_model), dtype=float)
            o = np.asarray(o, dtype=float)
            assert o.shape == ref.shape and o.ndim == 2
            np.testing.assert_allclose(o, ref, atol=1e-6)

