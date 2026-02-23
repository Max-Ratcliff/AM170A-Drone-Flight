"""Mission targets for generating waypoints and computing distances."""

import numpy as np
from numpy.random import default_rng


class Targets:
    """
    Generates and manages waypoint coordinates for drone mission planning.
    """

    def __init__(self, num_targets: int, bounds: tuple[float, float]) -> None:
        """
        Initialize the 2d environment.

        Args:
            num_targets: Number of target waypoints to generate.
            bounds: (min, max) for x and y coordinates (in meters).
        """
        self.num_targets = num_targets
        self.bounds = bounds
        self._rng = default_rng()

    def generate_waypoints(self) -> np.ndarray:
        """
        Generate N random (x, y) coordinates within the given bounds.

        Returns:
            N x 2 array of waypoint coordinates.
        """
        low, high = self.bounds
        waypoints = self._rng.uniform(low=low, high=high, size=(self.num_targets, 2))
        return waypoints.astype(np.float64)

    def get_distance_matrix(self, waypoints: np.ndarray) -> np.ndarray:
        """
        Compute the Euclidean distance between all waypoints.
        each entry (i, j) is the distance from waypoint i to j.

        Args:
            waypoints: N x 2 array of (x, y) coordinates.

        Returns:
            N x N symmetric matrix of pairwise distances (meters).
        """
        n = waypoints.shape[0]
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_matrix[i, j] = np.sqrt(
                        (waypoints[i, 0] - waypoints[j, 0]) ** 2
                        + (waypoints[i, 1] - waypoints[j, 1]) ** 2
                    )
        return dist_matrix
