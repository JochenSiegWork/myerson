"""Game-theoretic properties of the Myerson value: locality, equality with
the plain Shapley value on complete graphs, closed-form games, and the
null-player, linearity and relabeling-invariance axioms."""
import pytest
import numpy as np
import networkx as nx
from myerson import MyersonCalculator, ShapleyCalculator
from .myerson_helpers import (brute_force_myerson, make_weighted_worth, _weights_for, make_atsc0_worth, _reference_graphs, zagreb_m1_worth, wiener_index_worth, make_weighted_sq_worth, make_unanimity_worth, make_linear_combo_worth, _values_by_node)


class TestMyersonLocality:
    """The defining Myerson property: a node's value depends only on its own
    connected component (graph restriction), not on the rest of the graph.
    """

    def test_isolated_node_gets_its_standalone_worth(self):
        # An isolated node is always its own component, so its marginal is always
        # v({i}) and its Myerson value equals that standalone worth exactly.
        graph = nx.path_graph(3)
        graph.add_node(7)  # isolated
        worth = make_weighted_worth()  # v(S) = (sum node+1)^2
        calc = MyersonCalculator(graph=graph, coalition_function=worth)
        values = _values_by_node(calc, graph)
        assert values[7] == pytest.approx(worth((7,), graph), abs=1e-9)

    def test_adding_isolated_node_leaves_others_unchanged(self):
        base = nx.path_graph(4)
        worth = make_weighted_worth()
        base_vals = _values_by_node(
            MyersonCalculator(graph=base, coalition_function=worth), base)

        extended = base.copy()
        extended.add_node(99)  # disconnected extra player
        ext_vals = _values_by_node(
            MyersonCalculator(graph=extended, coalition_function=worth), extended)

        for node in base.nodes():
            assert ext_vals[node] == pytest.approx(base_vals[node], abs=1e-9), node

    def test_component_values_match_component_computed_alone(self):
        # Two disjoint components in one graph; a node's value must equal its
        # value when its component is computed in isolation.
        comp_a = nx.path_graph(3)                       # nodes 0,1,2
        comp_b = nx.relabel_nodes(nx.cycle_graph(3),
                                  {0: 10, 1: 11, 2: 12})  # nodes 10,11,12
        combined = nx.union(comp_a, comp_b)
        worth = make_weighted_worth()

        combined_vals = _values_by_node(
            MyersonCalculator(graph=combined, coalition_function=worth), combined)
        a_vals = _values_by_node(
            MyersonCalculator(graph=comp_a, coalition_function=worth), comp_a)
        b_vals = _values_by_node(
            MyersonCalculator(graph=comp_b, coalition_function=worth), comp_b)

        for node in comp_a.nodes():
            assert combined_vals[node] == pytest.approx(a_vals[node], abs=1e-9)
        for node in comp_b.nodes():
            assert combined_vals[node] == pytest.approx(b_vals[node], abs=1e-9)

class TestMyersonEqualsShapleyOnCompleteGraph:
    """On a complete graph every coalition is connected, so the graph
    restriction is a no-op and the Myerson value must equal the plain Shapley
    value. Cross-validates ``myerson.py`` against the independent ``shapley.py``.
    """

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    @pytest.mark.parametrize("worth_factory", [
        lambda: make_weighted_worth(),
        lambda: make_atsc0_worth(_weights_for(nx.complete_graph(5))),
        lambda: zagreb_m1_worth,
    ])
    def test_myerson_matches_shapley(self, n, worth_factory):
        graph = nx.complete_graph(n)
        worth = worth_factory()
        myerson = MyersonCalculator(
            graph=graph, coalition_function=worth).calculate_all_myerson_values()
        shapley = ShapleyCalculator(
            graph=graph, coalition_function=worth).calculate_all_shapley_values()
        assert np.allclose(myerson, shapley, atol=1e-9), f"{n=} {myerson=} {shapley=}"

class TestUnanimityGameClosedForm:
    """Exact rational closed form for the unanimity game."""

    @pytest.mark.parametrize("n,target", [
        (4, (0, 1)),
        (5, (0, 2, 4)),
        (5, (3,)),
        (6, (1, 2, 3, 4)),
    ])
    def test_complete_graph_matches_one_over_size(self, n, target):
        graph = nx.complete_graph(n)
        worth = make_unanimity_worth(target)
        values = MyersonCalculator(
            graph=graph, coalition_function=worth).calculate_all_myerson_values()
        expected = np.array(
            [1.0 / len(target) if node in target else 0.0
             for node in graph.nodes()])
        assert np.allclose(values, expected, atol=1e-9), f"{values=} {expected=}"

    def test_unanimity_on_path_matches_brute_force(self):
        # No closed form once connectivity restricts the coalitions, but the
        # brute-force oracle still applies.
        graph = nx.path_graph(5)
        worth = make_unanimity_worth((0, 4))
        got = MyersonCalculator(
            graph=graph, coalition_function=worth).calculate_all_myerson_values()
        expected = brute_force_myerson(graph, worth)
        assert np.allclose(got, expected, atol=1e-9), f"{got=} {expected=}"

class TestNullPlayerAxiom:
    """A player whose marginal contribution is always zero gets value zero."""

    def test_isolated_zero_worth_node_is_null(self):
        # Node 4 is isolated (never merges components) and has weight 0, so its
        # singleton worth is 0 -> it contributes nothing to any coalition, even
        # though the worth is non-additive for the connected players.
        graph = nx.path_graph(4)
        graph.add_node(4)
        weights = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 0.0}
        worth = make_weighted_sq_worth(weights)
        values = _values_by_node(
            MyersonCalculator(graph=graph, coalition_function=worth), graph)
        assert values[4] == pytest.approx(0.0, abs=1e-9)

class TestValueOperatorLinearity:
    """The Myerson value is linear in the coalition function/game."""

    @pytest.mark.parametrize("graph_name,graph", [
        ("path5", nx.path_graph(5)),
        ("cycle6", nx.cycle_graph(6)),
        ("gnp_7", nx.gnp_random_graph(7, 0.4, seed=2)),
    ])
    def test_additivity_and_scaling(self, graph_name, graph):
        v = make_weighted_worth()
        w = zagreb_m1_worth
        a, b = 2.5, -1.5

        mv_v = MyersonCalculator(
            graph=graph, coalition_function=v).calculate_all_myerson_values()
        mv_w = MyersonCalculator(
            graph=graph, coalition_function=w).calculate_all_myerson_values()
        mv_combo = MyersonCalculator(
            graph=graph,
            coalition_function=make_linear_combo_worth(a, v, b, w)
        ).calculate_all_myerson_values()

        assert np.allclose(mv_combo, a * mv_v + b * mv_w, atol=1e-9), (
            f"{graph_name}: {mv_combo=} {a*mv_v + b*mv_w=}")

class TestRelabelingInvariance:
    """Relabeling nodes must permute the Myerson values accordingly (a purely
    structural worth is used so the values themselves are label-independent).
    """

    @pytest.mark.parametrize("graph_name,graph", [
        ("path5", nx.path_graph(5)),
        ("ring_with_tail", _reference_graphs()["ring_with_tail"]),
        ("balanced_tree", nx.balanced_tree(2, 2)),
    ])
    def test_permuting_labels_permutes_values(self, graph_name, graph):
        worth = wiener_index_worth  # depends only on subgraph structure
        original = _values_by_node(
            MyersonCalculator(graph=graph, coalition_function=worth), graph)

        nodes = list(graph.nodes())
        offset = 100
        mapping = {node: node + offset for node in nodes}
        relabeled = nx.relabel_nodes(graph, mapping)
        relabeled_vals = _values_by_node(
            MyersonCalculator(graph=relabeled, coalition_function=worth), relabeled)

        for node in nodes:
            assert relabeled_vals[mapping[node]] == pytest.approx(
                original[node], abs=1e-9), f"{graph_name}: node {node}"
