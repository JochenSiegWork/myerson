"""`explain_batch` must match per-molecule `explain` (chemprop).
Verifies that the cross-molecule pooled-forward batch API returns the same
Myerson values as explaining each molecule individually — exactly for the exact
(small) molecules, and bit-identically for the sampled molecules under a fixed
seed. Covers both regression and classification models.
"""
import numpy as np
import pytest
import torch
from chemprop.models.model import MPNN
from chemprop.data import MoleculeDatapoint, MoleculeDataset
from myerson.chemprop_explain import (
    MyersonExplainer,
    MyersonSamplingExplainer,
    MyersonClassExplainer,
    MyersonSamplingClassExplainer,
    explain_batch,
)
CKPT = "tests/chemprop_regression.ckpt"
CLASS_CKPT = "tests/chemprop_multitask.ckpt"
SMILES = [
    "CCO",                       # ethanol  (N=3, exact)
    "c1ccccc1O",                 # phenol   (N=7, sampled with low threshold)
    "CC(=O)Oc1ccccc1C(=O)O",     # aspirin  (N=13, sampled)
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",# caffeine (N=14, sampled)
]
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
def _mg(smiles):
    return MoleculeDataset([MoleculeDatapoint.from_smi(smiles)])[0].mg
class TestExplainBatchMatchesPerMolecule:
    def test_mixed_exact_and_sampling(self, model):
        molgraphs = [_mg(s) for s in SMILES]
        threshold = 5  # ethanol exact; phenol/aspirin/caffeine sampled
        seed = 42
        n_samples = 200
        batched = explain_batch(
            molgraphs, model, seed=seed, number_of_samples=n_samples,
            sample_if_more_nodes_than=threshold)
        assert len(batched) == len(molgraphs)
        for i, (mg, smiles) in enumerate(zip(molgraphs, SMILES)):
            if mg.V.shape[0] > threshold:
                ref = MyersonSamplingExplainer(
                    mg, model, seed=seed, number_of_samples=n_samples
                ).sample_all_myerson_values()
            else:
                ref = MyersonExplainer(mg, model).calculate_all_myerson_values()
            np.testing.assert_allclose(
                np.asarray(batched[i], dtype=float),
                np.asarray(ref, dtype=float), atol=1e-6,
                err_msg=f"explain_batch != per-molecule for {smiles}")
    def test_all_sampled(self, model):
        molgraphs = [_mg(s) for s in SMILES]
        seed = 7
        n_samples = 150
        batched = explain_batch(
            molgraphs, model, seed=seed, number_of_samples=n_samples,
            sample_if_more_nodes_than=1)  # force every molecule to sample
        for i, mg in enumerate(molgraphs):
            ref = MyersonSamplingExplainer(
                mg, model, seed=seed, number_of_samples=n_samples
            ).sample_all_myerson_values()
            np.testing.assert_allclose(
                np.asarray(batched[i], dtype=float),
                np.asarray(ref, dtype=float), atol=1e-6)
class TestExplainBatchClassification:
    """`explain_batch(classification=True)` must match the per-molecule
    `*ClassExplainer`s (per-task tensor worths, shape (n_atoms, n_tasks))."""
    def test_mixed_exact_and_sampling(self, class_model):
        molgraphs = [_mg(s) for s in SMILES]
        threshold = 5  # ethanol exact; rest sampled
        seed = 42
        n_samples = 150
        batched = explain_batch(
            molgraphs, class_model, seed=seed, number_of_samples=n_samples,
            sample_if_more_nodes_than=threshold, classification=True)
        assert len(batched) == len(molgraphs)
        for i, (mg, smiles) in enumerate(zip(molgraphs, SMILES)):
            if mg.V.shape[0] > threshold:
                ref = MyersonSamplingClassExplainer(
                    mg, class_model, seed=seed, number_of_samples=n_samples
                ).sample_all_myerson_values()
            else:
                ref = MyersonClassExplainer(
                    mg, class_model).calculate_all_myerson_values()
            got = np.asarray(batched[i], dtype=float)
            ref = np.asarray(ref, dtype=float)
            assert got.shape == ref.shape and got.ndim == 2
            np.testing.assert_allclose(
                got, ref, atol=1e-6,
                err_msg=f"classification explain_batch != per-molecule for {smiles}")
