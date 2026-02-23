"""Aerodynamic power and energy models for drone flight segments."""

import numpy as np

from params import DroneParams


class DronePhysics:
    """Aerodynamic power and energy calculations for drone flight."""

    def __init__(self, params: DroneParams) -> None:
        """
        Initialize with physical constants.

        Args:
            params: Drone parameters (mass, coefficients, etc.).
        """
        self.params = params

    def calculate_power(self, v: float) -> float:
        """
        Compute aerodynamic power: P(v) = c1 + c2*v^3 + c3/v.

        Args:
            v: Flight velocity in m/s.

        Returns:
            Power in Watts.
        """
        c1, c2, c3 = self.params.c1, self.params.c2, self.params.c3
        return c1 + c2 * (v**3) + c3 / max(v, 1e-9)

    def calculate_energy(self, d: float, v: float) -> float:
        """
        Compute energy for a segment: E(v) = P(v) * d / v.

        Args:
            d: Segment distance in meters.
            v: Flight velocity in m/s.

        Returns:
            Energy in Joules. Returns inf if v <= 0.
        """
        if v <= 0:
            return float("inf")
        return self.calculate_power(v) * (d / v)
