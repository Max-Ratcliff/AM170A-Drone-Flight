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

    def find_optimal_velocity(self, distance: float) -> tuple[float, float]:
        """
        Find velocity v > 0.1 that minimizes segment energy.

        Args:
            distance: Segment length in meters.

        Returns:
            (optimal_velocity, minimum_energy) in (m/s, Joules).
        """

        def energy_func(v: float) -> float:
            return self.physics.calculate_energy(distance, v)

        result = minimize_scalar(
            energy_func,
            bounds=(0.1, 100.0),
            method="bounded",
        )
        return float(result.x), float(result.fun)

    def build_energy_matrix(
        self, distance_matrix: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Build N x N energy cost matrix from distances.

        Args:
            distance_matrix: N x N Euclidean distance matrix.

        Returns:
            (energy_matrix, universal_optimal_velocity). With zero wind, v_opt
            is constant for all segments.
        """
        n = distance_matrix.shape[0]
        energy_matrix = np.zeros((n, n))
        v_opt_universal = None

        for i in range(n):
            for j in range(n):
                if i == j:
                    energy_matrix[i, j] = 0.0
                else:
                    v_opt, e_min = self.find_optimal_velocity(distance_matrix[i, j])
                    energy_matrix[i, j] = e_min
                    if v_opt_universal is None:
                        v_opt_universal = v_opt

        return energy_matrix, float(v_opt_universal)

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
