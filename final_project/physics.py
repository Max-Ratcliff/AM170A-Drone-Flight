"""Aerodynamic power and energy models for drone flight segments."""

import numpy as np
from params import DroneParams


class DronePhysics:
    """Aerodynamic power and energy calculations for drone flight."""

    def __init__(
        self, params: DroneParams, wind_vector: tuple[float, float] = (0.0, 0.0)
    ) -> None:
        """
        Initialize with physical constants.

        Args:
            params: Drone parameters (mass, coefficients, etc.).
            wind_vector: Environmental wind (w_x, w_y) in m/s. Default is 0, 0.
        """
        self.params = params
        self.wind = np.array(wind_vector)

    def calculate_power(self, v: float) -> float:
        """
        Compute aerodynamic power: P(v) = c1 + c2*v^3 + c3/v.
        where v is the airspeed (ground speed adjusted for wind).
        c1: Blade profile power (constant)
        c2: Parasitic drag coefficient (scales with v^3)
        c3: Induced power coefficient (scales with 1/v)

        Args:
            v: Flight velocity in m/s.

        Returns:
            Power in Watts.
        """
        c1, c2, c3 = self.params.c1, self.params.c2, self.params.c3
        return c1 + c2 * (v**3) + c3 / max(v, 1e-9)

    def calculate_energy(self, segment_vector: np.ndarray, v_ground: float) -> float:
        """
        Compute energy for a segment: E(v) = P(v_air) * d / v_ground.

        Args:
            segment_vector: 2D vector of the flight segment in meters.
            v_ground: Ground speed in m/s.

        Returns:
            Energy in Joules. Returns inf if v_ground <= 0.
        """
        if v_ground <= 0:
            return float("inf")

        distance = float(np.linalg.norm(segment_vector))
        if distance == 0:
            return 0.0

        time = distance / v_ground

        # v_ground_vec points in the direction of the segment
        v_ground_vec = (segment_vector / distance) * v_ground
        # v_air = v_ground - wind
        v_air_vec = v_ground_vec - self.wind
        v_air = float(np.linalg.norm(v_air_vec))

        return self.calculate_power(v_air) * time
