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
            cluster_centers = self._rng.uniform(
                low=low, high=high, size=(num_clusters, 2)
            )
            waypoints = []
            for i in range(self.num_targets):
                center = cluster_centers[i % num_clusters]
                point = center + self._rng.normal(scale=(high - low) / 15, size=2)
                waypoints.append(np.clip(point, low, high))
            return np.array(waypoints)

        elif self.distribution == "grid":
            # Fit points to a grid
            grid_size = int(np.ceil(np.sqrt(self.num_targets)))
            x = np.linspace(
                low + (high - low) / 10, high - (high - low) / 10, grid_size
            )
            y = np.linspace(
                low + (high - low) / 10, high - (high - low) / 10, grid_size
            )
            xv, yv = np.meshgrid(x, y)
            grid_points = np.vstack([xv.ravel(), yv.ravel()]).T
            # Select random points from the grid and add minor jitter
            indices = self._rng.choice(
                len(grid_points), self.num_targets, replace=False
            )
            jitter = self._rng.normal(
                scale=(high - low) / 100, size=(self.num_targets, 2)
            )
            return grid_points[indices] + jitter

        # Default: Uniform with Minimum Distance Constraint
        # Ensures points aren't too close, which helps label legibility.
        waypoints = []
        min_dist = (high - low) / (self.num_targets ** 0.5 * 2.0) # Adaptive min distance
        
        max_attempts = 1000
        attempts = 0
        
        while len(waypoints) < self.num_targets and attempts < max_attempts:
            point = self._rng.uniform(low=low, high=high, size=2)
            
            # Check distance against all existing points
            is_valid = True
            for existing in waypoints:
                if np.linalg.norm(point - existing) < min_dist:
                    is_valid = False
                    break
            
            if is_valid:
                waypoints.append(point)
            attempts += 1
            
        # Fallback if constraint is too tight
        if len(waypoints) < self.num_targets:
            remaining = self.num_targets - len(waypoints)
            extra = self._rng.uniform(low=low, high=high, size=(remaining, 2))
            waypoints.extend(extra.tolist())

        return np.array(waypoints, dtype=np.float64)
