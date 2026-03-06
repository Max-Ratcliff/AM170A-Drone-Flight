"""Routing optimization: time optimization and TSP solution."""

import itertools
from typing import TYPE_CHECKING

import numpy as np
from physics import DronePhysics

if TYPE_CHECKING:
    pass


class RoutingOptimizer:
    """Finds optimal time per segment and solves TSP for minimum energy route."""

    def __init__(self, physics_model: DronePhysics) -> None:
        """
        Initialize with a drone physics model.

        Args:
            physics_model: Instance of DronePhysics for energy calculations.
        """
        self.physics = physics_model

    def build_energy_matrix(
        self, waypoints: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        """
        Build N x N energy cost matrix from waypoints.

        Args:
            waypoints: N x 2 array of waypoint coordinates.

        Returns:
            (energy_matrix, dictionary of optimal times per segment).
        """
        n = waypoints.shape[0]
        energy_matrix = np.zeros((n, n))
        optimal_times = {}

        for i in range(n):
            for j in range(n):
                if i == j:
                    energy_matrix[i, j] = 0.0
                else:
                    segment = waypoints[j] - waypoints[i]
                    res = self.physics.find_optimal_time(segment)
                    energy_matrix[i, j] = res.e_opt
                    optimal_times[(i, j)] = res.t_opt

        return energy_matrix, optimal_times

    def solve_tsp(self, cost_matrix: np.ndarray, method: str = "brute") -> list[int]:
        """
        Solve TSP: brute-force, nearest_neighbor, held_karp, or nn_2opt.
        """
        n = cost_matrix.shape[0]
        if n <= 1:
            return list(range(n))

        if method == "nearest_neighbor":
            return self._solve_nearest_neighbor(cost_matrix)
        if method == "held_karp":
            return self._solve_held_karp(cost_matrix)
        if method == "nn_2opt":
            init = self._solve_nearest_neighbor(cost_matrix)
            improved = self._two_opt(cost_matrix, init)
            return improved
            
        return self._solve_brute(cost_matrix)

    def _solve_brute(self, cost_matrix: np.ndarray) -> list[int]:
        """Brute-force TSP via itertools.permutations."""
        n = cost_matrix.shape[0]
        best_order: list[int] = []
        best_cost = float("inf")

        for perm in itertools.permutations(range(1, n)):
            order = [0] + list(perm)
            cost = self._tour_cost(cost_matrix, order)
            if cost < best_cost:
                best_cost = cost
                best_order = order

        return best_order

    def _solve_nearest_neighbor(self, cost_matrix: np.ndarray) -> list[int]:
        """Greedy nearest-neighbor heuristic."""
        n = cost_matrix.shape[0]
        order = [0]
        unvisited = set(range(1, n))

        while unvisited:
            current = order[-1]
            best_next = min(
                unvisited,
                key=lambda j: cost_matrix[current, j],
            )
            order.append(best_next)
            unvisited.remove(best_next)

        return order

    def _solve_held_karp(self, cost_matrix: np.ndarray) -> list[int]:
        """Exact TSP via Held-Karp dynamic programming."""
        n = cost_matrix.shape[0]
        memo = {}

        for i in range(1, n):
            memo[(frozenset([i]), i)] = (cost_matrix[0, i], 0)

        for subset_size in range(2, n):
            for subset in itertools.combinations(range(1, n), subset_size):
                subset_fs = frozenset(subset)
                for next_node in subset:
                    prev_subset = subset_fs - {next_node}
                    min_cost = float('inf')
                    min_prev_node = None
                    for prev_node in prev_subset:
                        cost = memo[(prev_subset, prev_node)][0] + cost_matrix[prev_node, next_node]
                        if cost < min_cost:
                            min_cost = cost
                            min_prev_node = prev_node
                    memo[(subset_fs, next_node)] = (min_cost, min_prev_node)

        all_nodes_fs = frozenset(range(1, n))
        min_cost = float('inf')
        last_node = None
        for node in range(1, n):
            cost = memo[(all_nodes_fs, node)][0] + cost_matrix[node, 0]
            if cost < min_cost:
                min_cost = cost
                last_node = node

        path = []
        curr_node = last_node
        curr_subset = all_nodes_fs
        while curr_node is not None and curr_node != 0:
            path.append(curr_node)
            _, prev_node = memo[(curr_subset, curr_node)]
            curr_subset = curr_subset - {curr_node}
            curr_node = prev_node
        path.append(0)
        path.reverse()
        return path

    def _tour_cost(self, cost_matrix: np.ndarray, order: list[int]) -> float:
        """Cycle cost: order[0] -> ... -> order[-1] -> order[0]."""
        total = 0.0
        for k in range(len(order)):
            i = order[k]
            j = order[(k + 1) % len(order)]
            total += float(cost_matrix[i, j])
        return total

    def _two_opt(self, cost_matrix: np.ndarray, order: list[int]) -> list[int]:
        """
        Standard 2-opt local search for a Hamiltonian cycle.
        Keeps node 0 as the fixed start (order[0] == 0).
        """
        n = len(order)
        if n < 4:
            return order[:]

        best = order[:]
        best_cost = self._tour_cost(cost_matrix, best)

        improved = True
        while improved:
            improved = False
            for i in range(1, n - 2):
                for k in range(i + 1, n - 1):
                    new_order = best[:]
                    new_order[i : k + 1] = reversed(new_order[i : k + 1])

                    new_cost = self._tour_cost(cost_matrix, new_order)
                    if new_cost < best_cost:
                        best = new_order
                        best_cost = new_cost
                        improved = True
                        break
                if improved:
                    break

        return best
