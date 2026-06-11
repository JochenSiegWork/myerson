import math
import networkx as nx
import numpy as np
from itertools import combinations, chain

from typing import Callable 

from tqdm import tqdm
import logging


# Byte-wise popcount lookup table, used to count coalition sizes for many
# integer-bitmask coalitions at once (vectorised exact Myerson computation).
_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def _popcount(masks: np.ndarray) -> np.ndarray:
    """Vectorised population count (number of set bits) for an integer array."""
    a = masks.astype(np.uint64, copy=True)
    counts = np.zeros(masks.shape, dtype=np.int64)
    for _ in range(8):  # 8 bytes covers up to 64-bit masks
        counts += _POPCOUNT_TABLE[(a & np.uint64(0xFF)).astype(np.uint8)]
        a >>= np.uint64(8)
    return counts



class MyersonCalculator():
    r"""Calculates the exact Myerson values. 
        For a game described by a coalition function :math:`v` the Myerson values
        attribute the individual players contribution to the payoff of the game. For
        a complete graph (every node connected to every other node) the Myerson
        value is equal to the Shapley value (:math:`S`: coalition of players,
        :math:`N`: grand coalition of all players):

        .. math::

            \text{Sh}_i\,({v}) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! \: (|N| - |S| - 1)!}{|N|!}\big( {v}\,(S \cup \{i\}) - {v}\,(S) \big)

        Else, the additional gain players can obthain through coalition is
        restricted only to players which are connected by edges in the graph. 

    Args:
        graph (nx.classes.graph.Graph): The coalition structure of the game.
        coalition_function (Callable): The coalition for which to calculate
            the payoff of the game. Expects a coalition (tuple of node
            indices) and a graph which contains additional information on
            the players to decide on the payoff. 

        disable_tqdm (bool, optional): Disables progress bar. Defaults to
            True.
    """

    def __init__(self,
                 graph: nx.classes.graph.Graph,
                 coalition_function: Callable,
                 disable_tqdm: bool=True) -> None:
        """Instantiate the class.
        """

        self.disable_tqdm = disable_tqdm
        self.log = logging.getLogger("MyersonCalculator")
        self.nx_graph = graph
        self.grand_coalition = list(graph.nodes()) # alias: set of players / set of nodes / F
        self.coalition_function = coalition_function

    def calculate_coalitions(self, grand_coalition: list) -> list:
        r"""Calculate all possible coalitions for a set of players / all nodes in
        a graph.

        Args:
            grand_coalition (list): Set of players / grand coalition / atoms in
                graph as a list of tupels.

        Returns:
            list: All :math:`2^N` coalitions.
        """
        self.log.info(f"Calculating number of coalitions.")
        coalitions = [combinations(grand_coalition, len(grand_coalition)-x) for x in \
                      tqdm(range(len(grand_coalition)), desc="Calculate coalitions", disable=self.disable_tqdm)]
        coalitions = list(chain.from_iterable(coalitions)) # chaining removes empty set
        coalitions.append(())
        self.log.info(f"Number of coalitions: {len(coalitions)}")
        return coalitions

    def calculate_graph_restricted_coalitions(self, coalitions: list, 
                                  nx_graph: nx.classes.graph.Graph) -> tuple[set, dict]:
        """Calculate the graph restricted coalitions for each coalition. The 
        graph restricted coalitions are tuples of nodes which are connected.

        Args:
            coalitions (list): All coalitions.
            nx_graph (nx.classes.graph.Graph): NetworkX Graph for which to
            calculate the Myerson values.

        Returns:
            tuple[set, dict]: Set of all possible graph restricted coalitions as
                a tuple of nodes, dictionary mapping each coalition to its
                graph restricted coalitions.
        """
        self.log.info(f"Calculating number of graph restricted coalitions.")
        graph_restricted_coalitions = [self.restrict(coalition, nx_graph) \
                                       for coalition in tqdm(coalitions,
                                           desc="Calculate graph restricted coalitions",
                                           disable=self.disable_tqdm)]
        coalitions_to_graph_restricted_coalitions = dict(zip(coalitions, graph_restricted_coalitions))
        graph_restricted_coalitions = list(chain.from_iterable(graph_restricted_coalitions))
        self.log.info(f"Removing dublicates from {len(graph_restricted_coalitions)} graph restricted coalitions.")
        graph_restricted_coalitions = set(x for x in tqdm(graph_restricted_coalitions, desc="Remove duplicates", disable=self.disable_tqdm))
        self.log.info(f"Number of graph restricted coalitions: {len(graph_restricted_coalitions)}")
        return graph_restricted_coalitions, coalitions_to_graph_restricted_coalitions

    def calculate_worth_of_single_graph_restricted_coalition(self,
        graph_restricted_coalition: tuple,
        nx_graph: nx.classes.graph.Graph) -> float:
        """Calculate the worth of a graph restricted coalition, i. e. a single
        connected component.

        Args:
            graph_restricted_coalition (tuple): Graph restricted coalition as
                node indices.
            nx_graph (nx.classes.graph.Graph): Additional information for the
                coalition function, i.e. the entire graph with node parameters. 
                The result of the coalition function should depend only on the
                coalition, however the node parameters might contain necessary 
                information.

        Returns:
            float: Worth, the output of the coalition function for the connected
            subgraph. 
        """
        return self.coalition_function(graph_restricted_coalition, nx_graph)

    def calculate_worth_of_graph_restricted_coalitions(self,
        graph_restricted_coalitions: set) -> dict:
        """Calculate the worth of every graph restricted coalition and map it to
        its worth. 

        Args:
            graph_restricted_coalitions (set): Set of connected components as
                tuples of node indices.

        Returns:
            dict: Dictionary mapping each connected component to its worth.
        """
        self.log.info(f"Calculating worth of graph restricted coalitions.")
        graph_restricted_coalitions_to_worth = {}
        for coalition in tqdm(graph_restricted_coalitions,
                              desc="Calculating worth of graph restricted coalitions",
                              disable=self.disable_tqdm):
            worth = self.calculate_worth_of_single_graph_restricted_coalition(coalition,
                                                                              self.nx_graph)
            graph_restricted_coalitions_to_worth.update({coalition: worth})
        return graph_restricted_coalitions_to_worth

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
            worth = 0.
            for graph_restricted_coalition in coalitions_to_graph_restricted_coalitions[coalition]:
                worth += graph_restricted_coalitions_to_worth[graph_restricted_coalition]
            coalition_to_worth.update({coalition: worth})
        return coalition_to_worth

    def calculate_worth_of_grand_coalition(self, nx_graph: nx.classes.graph.Graph) -> float:
        """Calculate payoff of the game, i.e. the payoff of all players / the
        grand coalition.

        Args:
            coalition_function (Callable): The coalition function associating a
                coalition with a payoff.
            nx_graph (nx.classes.graph.Graph): Coalition structure of the game
                as a graph.

        Returns:
            float: Payoff of the game / worth of grand coalition.
        """
        restricted_grand_coalition = self.restrict(self.grand_coalition, nx_graph)
        worth = sum([self.calculate_worth_of_single_graph_restricted_coalition(S, nx_graph) \
                     for S in restricted_grand_coalition])
        return worth 

    def _get_adjacency_masks(self, nx_graph: nx.classes.graph.Graph):
        """Build (and cache) a bitmask adjacency representation of ``nx_graph``.

        Each node is assigned an integer index ``0..N-1`` (in node iteration
        order). ``neighbor_masks[i]`` is an integer whose set bits are the
        indices of the neighbours of node ``i``. This lets ``restrict`` compute
        connected components with pure-integer bit operations, avoiding the
        per-call overhead of building NetworkX subgraph views.

        The cache is keyed on the identity of the graph object, so it is rebuilt
        automatically if a different graph is passed.

        Returns:
            tuple: ``(index_to_label, label_to_index, neighbor_masks)``.
        """
        if getattr(self, "_adj_graph_id", None) == id(nx_graph):
            return (self._adj_index_to_label,
                    self._adj_label_to_index,
                    self._adj_neighbor_masks)

        label_to_index = {label: i for i, label in enumerate(nx_graph.nodes())}
        index_to_label = list(label_to_index.keys())
        neighbor_masks = [0] * len(index_to_label)
        for u, v in nx_graph.edges():
            iu = label_to_index[u]
            iv = label_to_index[v]
            if iu == iv:
                continue  # ignore self-loops
            neighbor_masks[iu] |= (1 << iv)
            neighbor_masks[iv] |= (1 << iu)

        self._adj_graph_id = id(nx_graph)
        self._adj_label_to_index = label_to_index
        self._adj_index_to_label = index_to_label
        self._adj_neighbor_masks = neighbor_masks
        return index_to_label, label_to_index, neighbor_masks

    def number_connected_components(self, nx_graph: nx.classes.graph.Graph | None = None) -> int:
        """Count the connected components of ``nx_graph`` (default: ``self.nx_graph``).

        Uses the cached integer-bitmask adjacency (see :meth:`_get_adjacency_masks`)
        rather than a NetworkX graph algorithm, so it has no extra dependency on
        NetworkX traversal routines.

        Args:
            nx_graph (nx.classes.graph.Graph, optional): Graph to inspect.
                Defaults to ``self.nx_graph``.

        Returns:
            int: Number of connected components.
        """
        if nx_graph is None:
            nx_graph = self.nx_graph
        index_to_label, _, _ = self._get_adjacency_masks(nx_graph)
        grand_coalition = tuple(index_to_label)
        if not grand_coalition:
            return 0
        return len(self.restrict(grand_coalition, nx_graph))

    def restrict(self, coalition: tuple, nx_graph: nx.classes.graph.Graph) -> list[tuple]:
        """Restricts a graph through a (sub)set of nodes / players. Generate a
        list of graph restricted coalitions, i. e. a list of node indices of
        connected nodes in the subgraph.

        Connected components are computed with integer-bitmask BFS over a cached
        adjacency representation (see :meth:`_get_adjacency_masks`), which is
        substantially faster than constructing a NetworkX subgraph view per
        coalition. Each returned tuple lists its nodes in ascending index order,
        which is deterministic so the tuples are stable dictionary keys.

        Args:
            coalition (tuple): Nodes that remain in the graph.
            nx_graph (nx.classes.graph.Graph): Graph from which to generate
                subgraphs.

        Returns:
            list[tuple]: Graph restricted coalitions as tuples of node indices.
        """
        if not coalition:
            return [()] # empty_graph

        index_to_label, label_to_index, neighbor_masks = \
            self._get_adjacency_masks(nx_graph)

        # Bitmask of the nodes that still need to be assigned to a component.
        remaining = 0
        for label in coalition:
            remaining |= (1 << label_to_index[label])

        components = []
        while remaining:
            # Grow a component starting from the lowest remaining node.
            frontier = remaining & (-remaining)
            component = 0
            while frontier:
                component |= frontier
                neighbours = 0
                f = frontier
                while f:
                    low = f & (-f)
                    neighbours |= neighbor_masks[low.bit_length() - 1]
                    f ^= low
                # Only follow edges that stay inside the coalition.
                frontier = neighbours & remaining & ~component
            remaining &= ~component

            # Expand the component bitmask back into node labels.
            comp_labels = []
            c = component
            while c:
                low = c & (-c)
                comp_labels.append(index_to_label[low.bit_length() - 1])
                c ^= low
            components.append(tuple(comp_labels))

        return components


    def subgraph_from_coalition(self, graph_restricted_coalition: tuple,
                               nx_graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
        """Generates a subgraph from a graph restricted coalition (a subset of
        nodes / players) and a graph.

        Args:
            graph_restricted_coalition (tuple): Nodes / players which form the
                subgraph.
            nx_graph (nx.classes.graph.Graph): Subgraph induced in this graph by
                the nodes_set.

        Returns:
            nx.classes.graph.Graph: The new subgraph.
        """
        if len(graph_restricted_coalition) == 0:
            return nx.Graph()
        else:
            return nx_graph.subgraph(graph_restricted_coalition)

    def _precompute_prefactors(self, size_grand_coalition: int) -> list[float]:
        """Precompute Shapley prefactors for each coalition size.

        Args:
            size_grand_coalition (int): Number of players.

        Returns:
            list[float]: Prefactor for coalition size s at index s.
        """
        n_fact = math.factorial(size_grand_coalition)
        return [
            math.factorial(s) * math.factorial(size_grand_coalition - s - 1) / n_fact
            for s in range(size_grand_coalition)
        ]

    def calculate_single_myerson_value(self, node: int, grand_coalition: tuple,
                                  coalitions: list[tuple], coalition_to_worth: dict) -> float:
        """Calculate a single Myerson value.

        Args:
            node (int): Node index for which to calculate the Myerson value.
            grand_coalition (tuple): Set of all players.
            coalitions (list[tuple]): List of all coalitions.
            coalition_to_worth (dict): Mapping of every coalition to its worth.

        Returns:
            float: Myerson value.
        """
        my = 0
        size_grand_coalition = len(grand_coalition)
        prefactors = self._precompute_prefactors(size_grand_coalition)
        for coalition in coalitions:
            if node in coalition:
                continue
            size_coalition = len(coalition)
            worth_of_coalition = coalition_to_worth[coalition]
            worth_of_coalition_with_node = coalition_to_worth[tuple(sorted(coalition+(node,)))]
            my += prefactors[size_coalition] * (worth_of_coalition_with_node - worth_of_coalition)
        return my

    def _try_vectorized_myerson_values(self) -> np.ndarray | None:
        """Vectorised fast path for the exact Myerson values using integer
        bitmasks as coalition keys.

        Each coalition is encoded as an integer bitmask (bit ``i`` set iff
        ``grand_coalition[i]`` is in the coalition), so the worths can be stored
        in a flat NumPy array indexed by mask. The Myerson value of every node is
        then a single vectorised dot product over all coalitions not containing
        that node, avoiding the per-coalition ``tuple(sorted(...))`` re-keying of
        :meth:`calculate_single_myerson_value`.

        Returns:
            np.ndarray | None: The Myerson values (in ``grand_coalition`` order),
            or ``None`` to signal the caller to use the generic per-node loop.
            ``None`` is returned unless the worths are scalar and all ``2^N``
            coalitions are present, i.e. this is skipped for the sampler (partial
            coalition set) and the multi-output classifier (tensor worths).
        """
        n = len(self.grand_coalition)
        if n == 0 or len(self.coalitions_to_worth) != (1 << n):
            return None
        sample = next(iter(self.coalitions_to_worth.values()))
        if not isinstance(sample, (int, float, np.integer, np.floating)):
            return None

        bit_index = {label: i for i, label in enumerate(self.grand_coalition)}
        worth_by_mask = np.zeros(1 << n, dtype=np.float64)
        for coalition, worth in self.coalitions_to_worth.items():
            mask = 0
            for label in coalition:
                mask |= 1 << bit_index[label]
            worth_by_mask[mask] = worth

        return self._vectorized_myerson_from_worth_array(worth_by_mask, n)

    def _vectorized_myerson_from_worth_array(self, worth_by_mask: np.ndarray,
                                             n: int) -> np.ndarray:
        """Compute every Myerson value from a flat ``worth_by_mask`` array.

        For node ``i`` (bit ``i``) the Myerson value is
        ``Σ_S γ(|S|) (worth[S∪i] - worth[S])`` over all coalitions ``S`` not
        containing ``i``; this is a single vectorised dot product per node.

        Args:
            worth_by_mask (np.ndarray): ``worth_by_mask[m]`` is the worth of the
                coalition encoded by bitmask ``m`` (length ``2^n``).
            n (int): Number of players.

        Returns:
            np.ndarray: Myerson values in ``grand_coalition`` order.
        """
        sizes = _popcount(np.arange(1 << n, dtype=np.int64))  # sizes[m] == |m|
        prefactors = np.asarray(self._precompute_prefactors(n), dtype=np.float64)

        # Reshape into an n-dimensional hypercube so "node i present/absent" is a
        # plain slice along one axis (bit i ↔ axis n-1-i in C order). This avoids
        # building a 2^N boolean mask + fancy-index gather for every node.
        worth_nd = worth_by_mask.reshape((2,) * n)
        sizes_nd = sizes.reshape((2,) * n)

        my_values = np.empty(n, dtype=np.float64)
        for i in range(n):
            axis = n - 1 - i
            without = np.take(worth_nd, 0, axis=axis)          # bit i = 0
            with_node = np.take(worth_nd, 1, axis=axis)        # bit i = 1
            sizes_without = np.take(sizes_nd, 0, axis=axis)
            delta = (with_node - without).ravel()
            weights = prefactors[sizes_without.ravel()]
            my_values[i] = np.dot(weights, delta)
        return my_values

    def _calculate_all_myerson_values_subset_dp(self) -> np.ndarray:
        """Exact Myerson values via a lowest-set-bit subset-lattice DP.

        Instead of recomputing the connected components of ``G[S]`` from scratch
        for each of the ``2^N`` coalitions, this reuses the partition of
        ``S \\ {min(S)}``. With ``i = min(S)`` and ``R = S \\ {i}``::

            low_comp[S] = {i} ∪ ⋃ { c ∈ components(R) : c touches a neighbour of i }
            worth[S]    = v(low_comp[S]) + worth[S \\ low_comp[S]]

        ``components(R)`` is recovered by "peeling" the memoised ``low_comp``
        (valid because ``R < S``). A component of ``R`` merges into ``i``'s
        component iff it directly touches ``i`` (the components of ``R`` are
        pairwise non-adjacent, so there is no transitive chaining). The distinct
        ``low_comp`` values are exactly the connected induced subgraphs, so the
        (possibly neural-network) coalition function is evaluated only on those,
        in one batched call.

        This unifies ``restrict`` and ``map_coalition_to_worth`` into a single
        pass and is used for the exact, scalar-worth case only.

        Returns:
            np.ndarray: Myerson values in ``grand_coalition`` order.
        """
        n = len(self.grand_coalition)
        if n == 0:
            return np.array([], dtype=np.float64)

        index_to_label, label_to_index, neighbor_masks = \
            self._get_adjacency_masks(self.nx_graph)
        size = 1 << n

        # --- Pass 1: connected component of the lowest vertex for every subset.
        low_comp = [0] * size
        unique_component_masks = set()
        for S in range(1, size):
            b = S & (-S)
            i = b.bit_length() - 1
            neighbours_of_i = neighbor_masks[i]
            component = b
            rest = S ^ b
            while rest:                       # peel partition(R) via low_comp
                c = low_comp[rest]
                if c & neighbours_of_i:       # c is adjacent to i -> merge it in
                    component |= c
                rest ^= c
            low_comp[S] = component
            unique_component_masks.add(component)

        # --- Worth of each unique connected component (batched for NN models).
        unique_masks = list(unique_component_masks)
        component_tuples = []
        for mask in unique_masks:
            labels = []
            c = mask
            while c:
                low = c & (-c)
                labels.append(index_to_label[low.bit_length() - 1])
                c ^= low
            component_tuples.append(tuple(labels))

        worth_by_tuple = self.calculate_worth_of_graph_restricted_coalitions(
            component_tuples)
        worth_of_component = {mask: worth_by_tuple[t]
                              for mask, t in zip(unique_masks, component_tuples)}

        # Expose the connected subgraphs / their worths for introspection
        # (cheap: only #GRC entries, not 2^N).
        self.graph_restricted_coalitions = set(component_tuples)
        self.graph_restricted_coalitions_to_worth = worth_by_tuple

        # --- Pass 2: additive worth DP over the subset lattice.
        # Accumulate into a Python list (fast scalar reads/writes) and convert to
        # NumPy once at the end; per-element NumPy indexing in this 2^N loop is
        # markedly slower than plain list indexing.
        worth_list = [0.0] * size
        for S in range(1, size):
            c = low_comp[S]
            worth_list[S] = worth_of_component[c] + worth_list[S ^ c]
        worth_by_mask = np.asarray(worth_list, dtype=np.float64)
        self._worth_by_mask = worth_by_mask

        return self._vectorized_myerson_from_worth_array(worth_by_mask, n)

    def calculate_all_myerson_values(self) -> np.ndarray:
        """Calculate the Myerson values for every node / player in the graph.

        Returns:
            np.ndarray: Myerson values for each node.
        """
        if getattr(self, "_supports_subset_dp", True):
            my_values = self._calculate_all_myerson_values_subset_dp()
            log_string = "".join([f"\t{node}: {val}\n" for node, val
                                  in zip(self.grand_coalition, my_values)])
            self.log.info(f"Myerson Values:\n{log_string}")
            return my_values

        self.calculate_all_mappings()
        self.log.info(f"Calculating Myerson values.")
        my_values = self._try_vectorized_myerson_values()
        if my_values is None:
            my_values = []
            for node in tqdm(self.grand_coalition, desc="Calculating Myerson values.", disable=self.disable_tqdm):
                my_val = self.calculate_single_myerson_value(node, self.grand_coalition,
                                                      self.coalitions, self.coalitions_to_worth)
                my_values.append(my_val)
            my_values = np.array(my_values)
        log_string = "".join([f"\t{node}: {val}\n" for node, val in zip(self.grand_coalition, my_values)])
        self.log.info(f"Myerson Values:\n{log_string}")
        return my_values

    def calculate_all_mappings(self) -> None:
        """Calculates all coalitions, graph restricted coalitions, and their
        associated worths as class attributes:

            * `self.coalitions` (list[tuple])
            * `self.graph_restricted_coalitions` (set[tuple])
            * `self.coalitions_to_graph_restricted_coalitions` (dict)
            * `self.graph_restricted_coalitions_to_worth` (dict)
            * `self.coalitions_to_worth` (dict)
        """
        self.coalitions = self.calculate_coalitions(self.grand_coalition)

        self.graph_restricted_coalitions, self.coalitions_to_graph_restricted_coalitions \
            = self.calculate_graph_restricted_coalitions(self.coalitions, self.nx_graph)

        self.graph_restricted_coalitions_to_worth \
            = self.calculate_worth_of_graph_restricted_coalitions(self.graph_restricted_coalitions)

        self.coalitions_to_worth \
            = self.map_coalition_to_worth(self.coalitions, 
                                          self.coalitions_to_graph_restricted_coalitions,
                                          self.graph_restricted_coalitions_to_worth)

class MyersonSampler(MyersonCalculator):
    r"""A class approximating the Myerson value using Monte Carlo sampling.
        The Myerson values are approximated by randomly sampling from all  
        permutations needed to calculate the Shapley value:

        .. math::

            \text{Sh}_i\,({v}) = \frac{1}{|N|!}\; \sum_R \big({v}\,(P_i^R \cup \{i\}) - {v}(P_i^R)\big)

        For efficiencies sake, the sampled permutations are transformed into
        the corresponding coalitions.

    Args:
        graph (nx.classes.graph.Graph): The coalition structure of the game.
        coalition_function (Callable): The coalition for which to calculate
            the payoff of the game. Expects a coalition (tuple of node
            indices) and a graph which contains additional information on
            the players to decide on the payoff. 
        seed (None | int, optional): Seed for randomness. Defaults to None.
        number_of_samples (int, optional): Number of sampling steps. Defaults to 1000.
        disable_tqdm (bool, optional): Disables progress bar. Defaults to True.
    """
    def __init__(self,
                 graph: nx.classes.graph.Graph,
                 coalition_function: Callable,
                 seed: None | int = None, 
                 number_of_samples: int = 1000,
                 disable_tqdm: bool=True) -> None:

        self.disable_tqdm = disable_tqdm
        self.log = logging.getLogger("MyersonSampler")
        self.nx_graph = graph
        self.grand_coalition = list(graph.nodes()) # alias: set of players / set of nodes / F
        self.coalition_function = coalition_function
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.number_of_samples = number_of_samples
        """Instantiates the class.
        """

    @staticmethod
    def _replace_in_array(array: np.ndarray, value_to_replace, value_to_replace_with) -> np.ndarray:
        """Replace a value in an array with a different value.

        Args:
            array (np.ndarray): The array.
            value_to_replace (int): The value to replace.
            value_to_replace_with (int): The value to replace with.

        Returns:
            np.ndarray: The array with the replace value if it was found, else
                the original array.
        """
        # Convert to a flat array to work with a single loop for any dimension
        flat_array = array.ravel().copy()
        # Find the index of the first occurrence of value_to_replace
        index = np.where(flat_array == value_to_replace)[0]
        if index.size > 0:  # Check if the value was found
            flat_array[index[0]] = value_to_replace_with  # Replace the first occurrence
        # Reshape back to original array's shape
        return flat_array.reshape(array.shape)

    def reset_rng(self):
        """Reset random number generator to seed.
        """
        self.rng = np.random.default_rng(self.seed)

    def _sample_base_permutations(self, number_of_samples: int):
        """RNG-driven core of :meth:`sample_permutations`: draw the base
        prefix-coalitions, each excluding the randomly chosen pivot node.

        This isolates the random draws (so the optimized bitmask pipeline and
        the legacy :meth:`sample_permutations` produce identical samples for a
        given seed).

        Returns:
            tuple(int, list[np.ndarray]): the pivot node, and the sampled
            prefix-coalitions (without the pivot node).
        """
        nodes_array = np.array(self.grand_coalition)
        random_node = self.rng.choice(nodes_array)

        self.log.info(f"Sampling {number_of_samples} steps.")
        permutations_without_random_node = []
        for i in tqdm(range(number_of_samples),
                      desc="Sample permutations without random node",
                      disable=self.disable_tqdm):
            random_permutation_size: int = self.rng.integers(0, len(nodes_array))
            nodes_array_without_random_node = nodes_array.copy()
            indices_to_delete = np.where(nodes_array_without_random_node == random_node)
            nodes_array_without_random_node = np.delete(nodes_array_without_random_node,
                                                        indices_to_delete)
            sampled_permutation_without_random_node = self.rng.choice(nodes_array_without_random_node,
                                                        size=random_permutation_size,
                                                        replace=False)
            self.rng.shuffle(sampled_permutation_without_random_node)
            permutations_without_random_node.append(sampled_permutation_without_random_node)
        return random_node, permutations_without_random_node

    def sample_permutations(self, number_of_samples: int):
        """Uniformly sample permutations from all possible permutations. Samples
        `number_of_samples`*2*`number_of_nodes` permutations in total.

        Args:
            number_of_samples (int): How many sample steps should be carried out.

        Returns:
            tuple(int, list[np.ndarray], list[np.ndarray]): A randomly chosen
                node, the sampled permutations without the random node, all the
                sampled permutations.
        """
        random_node, permutations_without_random_node = \
            self._sample_base_permutations(number_of_samples)
        nodes_array = np.array(self.grand_coalition)

        all_sampled_permutations = []
        for permutation in tqdm(permutations_without_random_node,
                         desc="Sample permutations containing random node",
                         disable=self.disable_tqdm):
            for node_idx, node in enumerate(nodes_array):
                swapped = self._replace_in_array(permutation.copy(), node, random_node)
                all_sampled_permutations.append(swapped)
                all_sampled_permutations.append(np.append(swapped, node))
        # len(all_sampled_permutations): steps*2*len(self.grand_coalition)
        self.log.info(f"Sampled {len(all_sampled_permutations)} of {math.factorial(len(self.grand_coalition))} permutations.")

        return random_node, permutations_without_random_node, all_sampled_permutations

    def get_coalitions_from_permutations(self, permutations: list[np.ndarray]):
        """Get the set of coalitions from the different (sampled) permutations.

        Args:
            permutations (list[np.ndarray]): The permutations.

        Returns:
            list[tuple]: The sampled coalitions.
        """
        all_sampled_coalitions = list(set([tuple(np.sort(x)) for x in permutations]))
        self.log.info(f"Sampled {len(all_sampled_coalitions)} of {2**len(self.grand_coalition)} coalitions.")
        return all_sampled_coalitions

    def _build_sampling_worth_by_mask(self) -> dict:
        """Build the integer-bitmask machinery the estimator's value loop needs.

        From the sampled base coalitions (``self.permutations_without_random_node``)
        and the pivot (``self.random_node``) this computes:

            * ``self._node_bits`` — ``1 << i`` for each player ``i``.
            * ``self._random_node_bit`` — the pivot's bit.
            * ``self._base_masks`` — each base coalition as a bitmask.

        and the exact set of coalitions the estimator will query, whose worths
        are computed once (reusing the connected-component + worth machinery) and
        returned keyed by bitmask. The tuple-keyed attributes are also populated
        for backward compatibility / introspection.
        """
        nodes = list(self.grand_coalition)
        index = {label: i for i, label in enumerate(nodes)}
        node_bits = [1 << i for i in range(len(nodes))]
        random_node_bit = node_bits[index[self.random_node]]

        base_masks = []
        for perm in self.permutations_without_random_node:
            mask = 0
            for label in perm.tolist():
                mask |= node_bits[index[label]]
            base_masks.append(mask)

        # Exact set of coalitions the marginal-contribution estimator queries:
        # for each base coalition and each player j, the coalition with/without j
        # (j swapped for the pivot when already present — the sampling trick).
        needed = set()
        for mask in base_masks:
            for bit in node_bits:
                without = ((mask ^ bit) | random_node_bit) if (mask & bit) else mask
                needed.add(without)
                needed.add(without | bit)

        # Bitmask -> sorted node-label tuple, for the (label-based) worth code.
        mask_to_tuple = {}
        for mask in needed:
            labels = []
            m = mask
            while m:
                low = m & (-m)
                labels.append(nodes[low.bit_length() - 1])
                m ^= low
            mask_to_tuple[mask] = tuple(labels)

        coalitions = list(mask_to_tuple.values())
        self.coalitions = coalitions
        self.graph_restricted_coalitions, self.coalitions_to_graph_restricted_coalitions \
            = self.calculate_graph_restricted_coalitions(coalitions, self.nx_graph)
        self.graph_restricted_coalitions_to_worth \
            = self.calculate_worth_of_graph_restricted_coalitions(self.graph_restricted_coalitions)
        self.coalitions_to_worth \
            = self.map_coalition_to_worth(coalitions,
                                          self.coalitions_to_graph_restricted_coalitions,
                                          self.graph_restricted_coalitions_to_worth)

        self._node_bits = node_bits
        self._random_node_bit = random_node_bit
        self._base_masks = base_masks
        self._worth_by_mask = {mask: self.coalitions_to_worth[t]
                               for mask, t in mask_to_tuple.items()}
        return self._worth_by_mask

    def sample_all_mappings(self) -> None:
        """Samples the base permutations and the worths of every coalition the
        estimator will query, as class attributes:

            * `self.random_node` (int)
            * `self.permutations_without_random_node` (list[np.ndarray])
            * `self.coalitions` (list[tuple])
            * `self.graph_restricted_coalitions` (set[tuple])
            * `self.coalitions_to_graph_restricted_coalitions` (dict)
            * `self.graph_restricted_coalitions_to_worth` (dict)
            * `self.coalitions_to_worth` (dict)
            * bitmask helpers (`self._base_masks`, `self._node_bits`,
              `self._random_node_bit`, `self._worth_by_mask`)
        """
        self.random_node, self.permutations_without_random_node \
            = self._sample_base_permutations(self.number_of_samples)
        self._build_sampling_worth_by_mask()

    def sample_all_myerson_values(self) -> np.ndarray:
        """Use Monte Carlo sampling to approximate the Myerson values for every
        node / player in the graph.

        Returns:
            np.ndarray: Sampled Myerson values for each node.
        """
        self.sample_all_mappings()
        self.log.info(f"Calculating sampled Myerson values.")
        worth = self._worth_by_mask
        node_bits = self._node_bits
        random_node_bit = self._random_node_bit
        n = len(node_bits)

        # Integer-bitmask value loop: each iteration is a few int ops + two dict
        # lookups (no per-iteration array copy / sort / tuple construction).
        acc = [0.0] * n
        for mask in tqdm(self._base_masks, disable=self.disable_tqdm,
                         desc="Calculate sampled Myerson values"):
            for j in range(n):
                bit = node_bits[j]
                without = ((mask ^ bit) | random_node_bit) if (mask & bit) else mask
                acc[j] += worth[without | bit] - worth[without]

        my_values = np.array(acc, dtype=float) / self.number_of_samples
        log_string = "".join([f"\t{node}: {val:.4f}\n" for node, val in zip(self.grand_coalition, my_values)])
        self.log.info(f"Sampled Myerson Values:\n{log_string}")
        return my_values

    def calculate_all_myerson_values(self) -> None:
        """Not implemented for sampling class.

        Raises:
            NotImplementedError: The MyersonSampler only has the
                `sample_all_myerson_values()` method. To accuratly calculate the
                Myerson values, please use the `MyersonCalculator` class.
        """
        raise NotImplementedError("""The MyersonSampler only has the `sample_all_myerson_values()` method.
                     To accuratly calculate the Myerson values,
                     please use the `MyersonCalculator` class.""")