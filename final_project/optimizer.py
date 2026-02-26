"""Routing optimization using segment stop-at-waypoint energy."""

from __future__ import annotations

import itertools

import numpy as np

from physics import DronePhysics, SegmentResult


class RoutingOptimizer:
    """
    For each segment i->j:
      - compute optimal time T_ij that minimizes segment energy
      - energy_matrix[i,j] = E_min(d_ij)
    Then solve TSP over that energy matrix.
    """

    def __init__(self, physics_model: DronePhysics) -> None:
        self.physics = physics_model

    def find_optimal_time(self, distance: float) -> SegmentResult:
        return self.physics.find_optimal_time(distance)

    def build_energy_matrix(self, distance_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          energy_matrix: NxN energies
          time_matrix:   NxN optimal segment times (for debugging/plots)
        """
        n = distance_matrix.shape[0]
        energy_matrix = np.zeros((n, n), dtype=float)
        time_matrix = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(n):
                if i == j:
                    energy_matrix[i, j] = 0.0
                    time_matrix[i, j] = 0.0
                else:
                    seg = self.find_optimal_time(float(distance_matrix[i, j]))
                    energy_matrix[i, j] = seg.e_opt
                    time_matrix[i, j] = seg.t_opt

        return energy_matrix, time_matrix

    def solve_tsp(self, cost_matrix: np.ndarray, method: str = "brute") -> list[int]:
        n = cost_matrix.shape[0]
        if n <= 1:
            return list(range(n))

        if method == "nearest_neighbor":
            return self._solve_nearest_neighbor(cost_matrix)
        return self._solve_brute(cost_matrix)

    def _solve_brute(self, cost_matrix: np.ndarray) -> list[int]:
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
        n = cost_matrix.shape[0]
        order = [0]
        unvisited = set(range(1, n))

        while unvisited:
            cur = order[-1]
            nxt = min(unvisited, key=lambda j: cost_matrix[cur, j])
            order.append(nxt)
            unvisited.remove(nxt)

        return order