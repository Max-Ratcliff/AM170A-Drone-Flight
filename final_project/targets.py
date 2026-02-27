"""Mission targets for generating waypoints and computing distances."""

from typing import Optional

import numpy as np
from numpy.random import default_rng


class Targets:
    """
    Generates and manages waypoint coordinates for drone mission planning.
    """

    def __init__(
        self,
        num_targets: int,
        bounds: tuple[float, float],
        *,
        waypoint_set: Optional[list[tuple[float, float]]] = None,
        distribution: str = "uniform",
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the 2d environment.

        Args:
            num_targets: Number of target waypoints (used when waypoint_set is None).
            bounds: (min, max) for x and y coordinates (in meters).
            waypoint_set: Optional fixed waypoints for reproducibility.
            distribution: Type of spatial distribution ('uniform', 'clustered', 'grid').
            seed: Optional random seed for reproducible generation.
        """
        self.num_targets = num_targets
        self.bounds = bounds
        self.distribution = distribution
        self._waypoint_set = waypoint_set
        self._rng = default_rng(seed)

    def generate_waypoints(self) -> np.ndarray:
        """
        Return waypoints based on the selected distribution.

        Returns:
            N x 2 array of waypoint coordinates.
        """
        if self._waypoint_set is not None:
            return np.array(self._waypoint_set, dtype=np.float64)
        
        low, high = self.bounds
        
        if self.distribution == "clustered":
            # Generate 3 clusters
            num_clusters = 3
            cluster_centers = self._rng.uniform(low=low, high=high, size=(num_clusters, 2))
            waypoints = []
            for i in range(self.num_targets):
                center = cluster_centers[i % num_clusters]
                point = center + self._rng.normal(scale=(high-low)/15, size=2)
                waypoints.append(np.clip(point, low, high))
            return np.array(waypoints)
            
        elif self.distribution == "grid":
            # Fit points to a grid
            grid_size = int(np.ceil(np.sqrt(self.num_targets)))
            x = np.linspace(low + (high-low)/10, high - (high-low)/10, grid_size)
            y = np.linspace(low + (high-low)/10, high - (high-low)/10, grid_size)
            xv, yv = np.meshgrid(x, y)
            grid_points = np.vstack([xv.ravel(), yv.ravel()]).T
            # Select random points from the grid and add minor jitter
            indices = self._rng.choice(len(grid_points), self.num_targets, replace=False)
            jitter = self._rng.normal(scale=(high-low)/100, size=(self.num_targets, 2))
            return grid_points[indices] + jitter

        # Default: Uniform
        waypoints = self._rng.uniform(
            low=low, high=high, size=(self.num_targets, 2)
        )
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
