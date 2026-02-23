"""Routing optimization: velocity optimization and TSP solution."""

import itertools
from typing import TYPE_CHECKING

import numpy as np
from physics import DronePhysics
from scipy.optimize import minimize_scalar

if TYPE_CHECKING:
    pass


class RoutingOptimizer:
    """Finds optimal velocity per segment and solves TSP for minimum energy route."""

    def __init__(self, physics_model: DronePhysics) -> None:
        """
        Initialize with a drone physics model.

        Args:
            physics_model: Instance of DronePhysics for energy calculations.
        """
        self.physics = physics_model

    def find_optimal_velocity(self, segment_vector: np.ndarray) -> tuple[float, float]:
        """
        Find velocity v > 0.1 that minimizes segment energy.

        Args:
            segment_vector: 2D segment vector in meters.

        Returns:
            (optimal_velocity, minimum_energy) in (m/s, Joules).
        """
        distance = float(np.linalg.norm(segment_vector))
        if distance == 0:
            return 0.0, 0.0

        def energy_func(v: float) -> float:
            return self.physics.calculate_energy(segment_vector, v)

        result = minimize_scalar(
            energy_func,
            bounds=(0.1, 100.0),
            method="bounded",
        )
        return float(result.x), float(result.fun)

    def build_energy_matrix(
        self, waypoints: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        """
        Build N x N energy cost matrix from waypoints.

        Args:
            waypoints: N x 2 array of waypoint coordinates.

        Returns:
            (energy_matrix, dictionary of optimal velocities per segment).
        """
        n = waypoints.shape[0]
        energy_matrix = np.zeros((n, n))
        optimal_velocities = {}

        for i in range(n):
            for j in range(n):
                if i == j:
                    energy_matrix[i, j] = 0.0
                else:
                    segment = waypoints[j] - waypoints[i]
                    v_opt, e_min = self.find_optimal_velocity(segment)
                    energy_matrix[i, j] = e_min
                    optimal_velocities[(i, j)] = v_opt

        return energy_matrix, optimal_velocities

    def solve_tsp(self, cost_matrix: np.ndarray, method: str = "brute") -> list[int]:
        """
        Solve TSP: brute-force or nearest-neighbor heuristic.

        Args:
            cost_matrix: N x N symmetric cost (energy) matrix.
            method: "brute" for exhaustive search, "nearest_neighbor" for greedy.

        Returns:
            Ordered list of waypoint indices (closed loop: start=end).
        """
        n = cost_matrix.shape[0]
        if n <= 1:
            return list(range(n))

        if method == "nearest_neighbor":
            return self._solve_nearest_neighbor(cost_matrix)
        if method == "held_karp":
            return self._solve_held_karp(cost_matrix)
        return self._solve_brute(cost_matrix)

    def _solve_brute(self, cost_matrix: np.ndarray) -> list[int]:
        """Brute-force TSP via itertools.permutations."""
        n = cost_matrix.shape[0]
        best_order: list[int] = []
        best_cost = float("inf")

        for perm in itertools.permutations(range(1, n)):
            order = [0] + list(perm)
            cost = 0.0
            for k in range(len(order) - 1):
                cost += cost_matrix[order[k], order[k + 1]]
            cost += cost_matrix[order[-1], order[0]]

            if cost < best_cost:
                best_cost = cost
                best_order = order

        return best_order

    def _solve_nearest_neighbor(self, cost_matrix: np.ndarray) -> list[int]:
        """Greedy nearest-neighbor heuristic: start at 0, visit lowest-cost next."""
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
        # Memoization table: maps (frozenset of visited_nodes, last_node) -> (cost, previous_node)
        memo = {}

        # Initialize base cases: path from start (0) to each other node directly
        for i in range(1, n):
            memo[(frozenset([i]), i)] = (cost_matrix[0, i], 0)

        # Iterate over subset sizes
        for subset_size in range(2, n):
            for subset in itertools.combinations(range(1, n), subset_size):
                subset_fs = frozenset(subset)
                for next_node in subset:
                    # Previous subset without next_node
                    prev_subset = subset_fs - {next_node}
                    
                    min_cost = float('inf')
                    min_prev_node = None
                    
                    for prev_node in prev_subset:
                        cost = memo[(prev_subset, prev_node)][0] + cost_matrix[prev_node, next_node]
                        if cost < min_cost:
                            min_cost = cost
                            min_prev_node = prev_node
                            
                    memo[(subset_fs, next_node)] = (min_cost, min_prev_node)

        # Connect the last node back to the start (0)
        all_nodes_fs = frozenset(range(1, n))
        min_cost = float('inf')
        last_node = None
        
        for node in range(1, n):
            cost = memo[(all_nodes_fs, node)][0] + cost_matrix[node, 0]
            if cost < min_cost:
                min_cost = cost
                last_node = node

        # Backtrack to find the optimal path
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
