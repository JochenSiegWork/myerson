"""Myerson values of concrete coalition functions: additive/normalized
closed forms and non-additive cheminformatics-style descriptors."""
import pytest
import numpy as np
import networkx as nx
from myerson import MyersonCalculator
from .myerson_helpers import (brute_force_myerson, _weights_for, make_additive_worth, make_mass_fraction_worth, make_component_normalized_worth, make_atsc0_worth, make_sum_squared_masses_worth, make_sumsq_over_size_worth, _reference_graphs, wiener_index_worth, _TOPOLOGICAL_DESCRIPTORS)


class TestClosedFormCoalitionFunctions:
    """Coalition functions with a known analytic attribution. These pin the
    interpretability contract: for a linear model the Myerson values must
    recover the exact per-atom contributions, independent of graph topology.
    """

    @pytest.mark.parametrize("name,graph", [
        ("path5", nx.path_graph(5)),
        ("cycle6", nx.cycle_graph(6)),
        ("star4", nx.star_graph(4)),
        ("two_triangles_disconnected", nx.disjoint_union(
            nx.complete_graph(3), nx.complete_graph(3))),
        ("gnp_7", nx.gnp_random_graph(7, 0.4, seed=2)),
    ])
    def test_additive_worth_recovers_atom_weights(self, name, graph):
        # v(S) = sum of atom weights (molecular weight). The graph restriction
        # cancels for an additive game, so MV_i == w_i regardless of topology.
        weights = _weights_for(graph)
        calc = MyersonCalculator(
            graph=graph, coalition_function=make_additive_worth(weights))
        values = calc.calculate_all_myerson_values()
        expected = np.array([weights[node] for node in graph.nodes()])
        assert np.allclose(values, expected, atol=1e-9), (
            f"{name}: {values=} {expected=}")

    @pytest.mark.parametrize("name,graph", [
        ("path5", nx.path_graph(5)),
        ("two_triangles_disconnected", nx.disjoint_union(
            nx.complete_graph(3), nx.complete_graph(3))),
    ])
    def test_mass_fraction_worth_recovers_normalized_weights(self, name, graph):
        # Normalising by the (constant) molecule total keeps the game additive,
        # so MV_i == w_i / total and the values sum to 1.
        weights = _weights_for(graph)
        total = sum(weights.values())
        calc = MyersonCalculator(
            graph=graph,
            coalition_function=make_mass_fraction_worth(weights, total))
        values = calc.calculate_all_myerson_values()
        expected = np.array([weights[node] / total for node in graph.nodes()])
        assert np.allclose(values, expected, atol=1e-9), (
            f"{name}: {values=} {expected=}")
        assert values.sum() == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("name,graph", [
        ("path5", nx.path_graph(5)),
        ("cycle6", nx.cycle_graph(6)),
        ("gnp_7", nx.gnp_random_graph(7, 0.4, seed=2)),
    ])
    def test_component_normalized_worth_is_nonadditive(self, name, graph):
        # v'(S) = number of connected components of S: non-additive and
        # topology-dependent. No simple per-atom formula, so pin it with the
        # brute-force oracle and the efficiency axiom.
        worth = make_component_normalized_worth()
        calc = MyersonCalculator(graph=graph, coalition_function=worth)
        values = calc.calculate_all_myerson_values()
        expected = brute_force_myerson(graph, worth)
        assert np.allclose(values, expected, atol=1e-9), (
            f"{name}: {values=} {expected=}")
        # Efficiency: values sum to v'(N) = number of components of the graph.
        n_components = nx.number_connected_components(graph)
        assert values.sum() == pytest.approx(float(n_components), abs=1e-9)

class TestNonAdditiveDescriptors:
    """Realistic, size-dependent non-additive coalition functions (molecular
    descriptors) cross-checked against the brute-force reference. Complements the
    closed-form additive tests by stressing the weighting where marginal
    contributions can be negative and vary with coalition size.
    """

    @pytest.mark.parametrize("name,graph", [
        ("path5", nx.path_graph(5)),
        ("cycle6_benzene", nx.cycle_graph(6)),
        ("star4", nx.star_graph(4)),
        ("two_triangles_disconnected", nx.disjoint_union(
            nx.complete_graph(3), nx.complete_graph(3))),
        ("ring_with_tail", _reference_graphs()["ring_with_tail"]),
        ("gnp_7", nx.gnp_random_graph(7, 0.4, seed=2)),
    ])
    def test_atsc0_matches_brute_force(self, name, graph):
        worth = make_atsc0_worth(_weights_for(graph))
        calc = MyersonCalculator(graph=graph, coalition_function=worth)
        got = calc.calculate_all_myerson_values()
        expected = brute_force_myerson(graph, worth)
        assert np.allclose(got, expected, atol=1e-9), (
            f"{name}: {got=} {expected=}")

    def test_atsc0_efficiency_axiom(self):
        # Even for a non-additive descriptor the Myerson values must sum to the
        # worth of the (graph-restricted) grand coalition.
        graph = _reference_graphs()["ring_with_tail"]
        worth = make_atsc0_worth(_weights_for(graph))
        calc = MyersonCalculator(graph=graph, coalition_function=worth)
        values = calc.calculate_all_myerson_values()
        grand = calc.calculate_worth_of_grand_coalition(graph)
        assert values.sum() == pytest.approx(grand, abs=1e-9)

class TestNonAdditiveAxioms:
    """Node-by-node checks for the ATSC0 descriptor that hold *without* a
    closed-form solution: value-operator axioms (symmetry, linearity). These
    give independent per-node assertions where a hand-derived expected value is
    not tractable.
    """

    def test_symmetry_path_with_mirror_symmetric_masses(self):
        # Path 0-1-2-3-4 with masses mirror-symmetric about the centre. The map
        # k -> 4-k is a graph automorphism that also preserves masses, so nodes
        # 0&4 and 1&3 are interchangeable and must get equal Myerson values.
        graph = nx.path_graph(5)
        weights = {0: 12.0, 1: 16.0, 2: 14.0, 3: 16.0, 4: 12.0}
        calc = MyersonCalculator(
            graph=graph, coalition_function=make_atsc0_worth(weights))
        v = calc.calculate_all_myerson_values()
        assert v[0] == pytest.approx(v[4], abs=1e-9)
        assert v[1] == pytest.approx(v[3], abs=1e-9)

    def test_symmetry_cycle_all_equal_but_one(self):
        # On a 6-cycle, reflecting through node 0 and its opposite (node 3) is an
        # automorphism swapping 1<->5 and 2<->4. Masses chosen to respect it.
        graph = nx.cycle_graph(6)
        weights = {0: 20.0, 1: 12.0, 2: 16.0, 3: 30.0, 4: 16.0, 5: 12.0}
        calc = MyersonCalculator(
            graph=graph, coalition_function=make_atsc0_worth(weights))
        v = calc.calculate_all_myerson_values()
        assert v[1] == pytest.approx(v[5], abs=1e-9)
        assert v[2] == pytest.approx(v[4], abs=1e-9)

    @pytest.mark.parametrize("name,graph", [
        ("path5", nx.path_graph(5)),
        ("cycle6", nx.cycle_graph(6)),
        ("gnp_7", nx.gnp_random_graph(7, 0.4, seed=2)),
    ])
    def test_linearity_decomposition_reconciles_to_closed_form(self, name, graph):
        # ATSC0 = A - B with A(S)=sum m_i^2 (additive, MV_i = m_i^2) and
        # B(S)=(sum m_i)^2/|S|. By linearity of the Myerson value in the game:
        #     MV_i(ATSC0) + MV_i(B) == MV_i(A) == m_i^2   (exact hand value).
        weights = _weights_for(graph)
        mv_full = MyersonCalculator(
            graph=graph,
            coalition_function=make_atsc0_worth(weights)
        ).calculate_all_myerson_values()
        mv_b = MyersonCalculator(
            graph=graph,
            coalition_function=make_sumsq_over_size_worth(weights)
        ).calculate_all_myerson_values()
        expected_msq = np.array([weights[node] ** 2 for node in graph.nodes()])
        assert np.allclose(mv_full + mv_b, expected_msq, atol=1e-9), (
            f"{name}: {mv_full + mv_b=} {expected_msq=}")

        # And MV of the additive component alone must equal m_i^2 directly.
        mv_a = MyersonCalculator(
            graph=graph,
            coalition_function=make_sum_squared_masses_worth(weights)
        ).calculate_all_myerson_values()
        assert np.allclose(mv_a, expected_msq, atol=1e-9), f"{name}: {mv_a=}"

class TestTopologicalDescriptors:
    """Myerson values of common structure-only cheminformatics descriptors.

    These worths depend on degrees / shortest paths / eccentricity of the
    induced subgraph rather than on node weights, so as a coalition grows a
    node's marginal contribution shifts with the surrounding topology. There is
    no per-node closed form, so each is cross-checked against the independent
    ``2^N`` brute-force oracle across a range of topologies, plus the efficiency
    axiom.
    """

    @pytest.mark.parametrize("desc_name,worth",
                             list(_TOPOLOGICAL_DESCRIPTORS.items()))
    @pytest.mark.parametrize("graph_name,graph",
                             list(_reference_graphs().items()))
    def test_matches_brute_force(self, desc_name, worth, graph_name, graph):
        calc = MyersonCalculator(graph=graph, coalition_function=worth)
        got = calc.calculate_all_myerson_values()
        expected = brute_force_myerson(graph, worth)
        assert np.allclose(got, expected, atol=1e-9), (
            f"{desc_name} on {graph_name}: {got=} {expected=}")

    @pytest.mark.parametrize("desc_name,worth",
                             list(_TOPOLOGICAL_DESCRIPTORS.items()))
    def test_efficiency_axiom(self, desc_name, worth):
        graph = _reference_graphs()["ring_with_tail"]
        calc = MyersonCalculator(graph=graph, coalition_function=worth)
        values = calc.calculate_all_myerson_values()
        grand = calc.calculate_worth_of_grand_coalition(graph)
        assert values.sum() == pytest.approx(grand, abs=1e-9), desc_name

    def test_symmetry_wiener_on_symmetric_tree(self):
        # A star's leaves are all interchangeable (graph automorphisms), so every
        # leaf must receive the same Wiener Myerson value, distinct from the hub.
        graph = nx.star_graph(4)  # node 0 = hub, 1..4 = leaves
        calc = MyersonCalculator(graph=graph,
                                 coalition_function=wiener_index_worth)
        v = calc.calculate_all_myerson_values()
        leaves = [v[i] for i in range(1, 5)]
        assert np.allclose(leaves, leaves[0], atol=1e-9), f"{v=}"
        assert not np.isclose(v[0], leaves[0], atol=1e-9), f"hub==leaf? {v=}"
