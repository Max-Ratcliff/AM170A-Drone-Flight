"""Physics + energy model for stop-at-waypoint segments."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from params import DroneParams


@dataclass(frozen=True)
class SegmentResult:
    """Convenient bundle of optimal segment data."""
    distance: float
    t_opt: float
    e_opt: float


class DronePhysics:
    """
    Segment model:
      v(t) = alpha * t * (T - t) (scalar speed along the segment direction)
      alpha chosen so that total distance traveled equals d.

    def __init__(
        self, params: DroneParams, wind_vector: tuple[float, float] = (0.0, 0.0)
    ) -> None:
        """
        For v(t)=alpha t(T-t):
        distance = ∫0^T v(t) dt = alpha * T^3 / 6
        => alpha = 6d / T^3
        """
        return 6.0 * d / (T**3)

    @staticmethod
    def _v_profile(alpha: float, T: float, t: np.ndarray) -> np.ndarray:
        return alpha * t * (T - t)

    @staticmethod
    def _a_profile(alpha: float, T: float, t: np.ndarray) -> np.ndarray:
        # derivative of alpha*t(T-t) = alpha*(T - 2t)
        return alpha * (T - 2.0 * t)

    @staticmethod
    def _s_profile(alpha: float, T: float, t: np.ndarray) -> np.ndarray:
        # s(t) = ∫ v dt = alpha * (T t^2 / 2 - t^3 / 3)
        return alpha * (T * (t**2) / 2.0 - (t**3) / 3.0)

    # Energy
    def segment_energy(self, d: float, T: float) -> float:
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
        if d < 0 or T <= 0:
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
