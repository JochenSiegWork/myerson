"""Internal bitmask/connected-component helpers."""
import pytest
import numpy as np
import networkx as nx
from myerson import MyersonCalculator


class TestBitmaskHelpers:
    """Unit tests for the shared bitmask helpers used by ``restrict`` and the
    sampling connected-component enumeration.
    """

    def _adjacency(self, graph):
        # A trivial coalition function is enough; we only need the adjacency
        # machinery, not any worth evaluation.
        calc = MyersonCalculator(graph=graph,
                                 coalition_function=lambda coalition, g: 0.0)
        return calc, calc._get_adjacency_masks(graph)

    def test_connected_component_masks_splits_disconnected_subgraph(self):
        # L--R--R with an isolated node 4: indices 0=1, 1=2, 2=3, 3=4.
        graph = nx.Graph()
        graph.add_edges_from([(1, 2), (2, 3)])
        graph.add_node(4)
        calc, (index_to_label, label_to_index, neighbor_masks) = \
            self._adjacency(graph)

        full = 0b1111  # all four nodes
        comps = calc._connected_component_masks(full, neighbor_masks)
        # {1,2,3} -> bits 0,1,2 = 0b0111; {4} -> bit 3 = 0b1000. Lowest-first.
        assert comps == [0b0111, 0b1000]

    def test_connected_component_masks_respects_induced_subgraph(self):
        # {1,3} share no edge inside the coalition (their link is via 2).
        graph = nx.Graph()
        graph.add_edges_from([(1, 2), (2, 3)])
        _, (index_to_label, label_to_index, neighbor_masks) = \
            self._adjacency(graph)
        calc = MyersonCalculator(graph=graph,
                                 coalition_function=lambda coalition, g: 0.0)

        mask = (1 << label_to_index[1]) | (1 << label_to_index[3])
        comps = calc._connected_component_masks(mask, neighbor_masks)
        assert comps == [1 << label_to_index[1], 1 << label_to_index[3]]

    def test_connected_component_masks_empty_mask(self):
        graph = nx.path_graph(3)
        calc, (_, _, neighbor_masks) = self._adjacency(graph)
        assert calc._connected_component_masks(0, neighbor_masks) == []

    def test_mask_to_labels_expands_in_ascending_index_order(self):
        graph = nx.Graph()
        graph.add_edges_from([(1, 2), (2, 3)])
        graph.add_node(4)
        calc, (index_to_label, _, _) = self._adjacency(graph)

        assert calc._mask_to_labels(0, index_to_label) == ()
        assert calc._mask_to_labels(0b1000, index_to_label) == (4,)
        # Bits set out of order must still come out ascending by index.
        assert calc._mask_to_labels(0b0101, index_to_label) == (
            index_to_label[0], index_to_label[2])
        assert calc._mask_to_labels(0b1111, index_to_label) == tuple(index_to_label)

    def test_helpers_reconstruct_restrict(self):
        # restrict() must be exactly the composition of the two helpers.
        graph = nx.Graph()
        graph.add_edges_from([(1, 2), (2, 3), (3, 1), (4, 5)])
        graph.add_node(6)
        calc, (index_to_label, label_to_index, neighbor_masks) = \
            self._adjacency(graph)

        coalition = (1, 2, 4, 6)
        node_mask = 0
        for label in coalition:
            node_mask |= 1 << label_to_index[label]
        expected = [calc._mask_to_labels(c, index_to_label)
                    for c in calc._connected_component_masks(
                        node_mask, neighbor_masks)]
        assert calc.restrict(coalition, graph) == expected

    def test_restrict_matches_networkx_components(self):
        # Cross-check the rewired restrict against NetworkX on several coalitions.
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5)])
        graph.add_node(6)
        calc = MyersonCalculator(graph=graph,
                                 coalition_function=lambda coalition, g: 0.0)

        coalitions = [
            (0, 1, 2, 3, 4, 5, 6),
            (0, 2, 4, 5),
            (0, 3, 6),
            (1,),
            (),
        ]
        for coalition in coalitions:
            got = {frozenset(c) for c in calc.restrict(coalition, graph)}
            sub = graph.subgraph(coalition)
            expected = {frozenset(c) for c in nx.connected_components(sub)}
            if not coalition:
                expected = {frozenset()}
            assert got == expected, f"{coalition=}: {got=} {expected=}"
