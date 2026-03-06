"""Physics + energy model for stop-at-waypoint segments with wind support."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from params import DroneParams
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class SegmentResult:
    """Convenient bundle of optimal segment data."""

    distance: float
    t_opt: float
    e_opt: float


class DronePhysics:
    """
    Physics-grounded drone flight model for energy-optimized routing.

    This model implements a 'stop-at-waypoint' flight profile, where the drone
    must come to a complete stop at both the beginning and end of every segment.
    It uses a parabolic velocity profile to minimize mechanical power.

    Attributes:
        params (DroneParams): Physical constants (mass, drag, power) of the drone.
        wind (np.ndarray): 2D wind vector influencing aerodynamic drag.
    """

    def __init__(
        self, params: DroneParams, wind_vector: tuple[float, float] = (0.0, 0.0)
    ) -> None:
        """
        Initialize the physics engine.

        Args:
            params: Configuration for drone mass, drag coefficient, etc.
            wind_vector: (x, y) wind velocity in meters per second.
        """
        self.p = params
        self.wind = np.array(wind_vector)

    # Core kinematics (1D along the segment)
    @staticmethod
    def _alpha_for_distance(d: float, T: float) -> float:
        """
        Calculate the scaling factor alpha for the parabolic velocity profile.

        For v(t) = alpha * t * (T - t):
        distance = ∫0^T v(t) dt = alpha * T^3 / 6
        => alpha = 6d / T^3

        Args:
            d: Target distance for the segment (meters).
            T: Total travel time (seconds).

        Returns:
            Kinematic scaling factor alpha.
        """
        return 6.0 * d / (T**3)

    @staticmethod
    def _v_profile(alpha: float, T: float, t: np.ndarray) -> np.ndarray:
        """Parabolic velocity profile: v(t) = alpha * t * (T - t)."""
        return alpha * t * (T - t)

    @staticmethod
    def _a_profile(alpha: float, T: float, t: np.ndarray) -> np.ndarray:
        """Linear acceleration profile: a(t) = alpha * (T - 2t)."""
        return alpha * (T - 2.0 * t)

    @staticmethod
    def _s_profile(alpha: float, T: float, t: np.ndarray) -> np.ndarray:
        """Distance profile: s(t) = ∫ v dt = alpha * (T * t^2 / 2 - t^3 / 3)."""
        return alpha * (T * (t**2) / 2.0 - (t**3) / 3.0)

    # Energy
    def segment_energy(self, segment_vector: np.ndarray, T: float) -> float:
        """
        Compute total energy (Joules) for a segment executed in time T.

        Integrates (P_hover + |F_thrust(t) · v_ground(t)|) dt over [0, T].
        Includes aerodynamic drag and inertial forces.

        Args:
            segment_vector: 2D vector (dx, dy) of the segment.
            T: Travel time (seconds).

        Returns:
            Total energy consumption in Joules. Returns +inf if T is non-positive.
        """
        if T <= 0:
            return float("inf")

        d = float(np.linalg.norm(segment_vector))
        if d == 0:
            # If no movement, just hover for T seconds
            return self.p.hover_power * T

        u = segment_vector / d
        w_parallel = np.dot(self.wind, u)

        alpha = self._alpha_for_distance(d, T)

        n = max(20, int(self.p.integration_steps))
        t = np.linspace(0.0, T, n)

        v_g = self._v_profile(alpha, T, t)  # ground speed scalar
        a = self._a_profile(alpha, T, t)  # acceleration scalar

        # Thrust component along segment: m*a + C*(v_g - w_parallel)
        # We assume power = hover_power + |F_thrust_parallel * v_g|
        f_thrust_parallel = self.p.mass * a + self.p.drag_coeff * (v_g - w_parallel)

        power_thrust = np.abs(f_thrust_parallel * v_g)
        power_total = self.p.hover_power + power_thrust

        return float(np.trapz(power_total, t))

    # Bounds + optimization over T
    def feasible_time_bounds(self, d: float) -> tuple[float, float]:
        """
        Compute conservative search bounds [T_low, T_high] for travel time T.

        Bounds are derived from peak velocity (v_max) and peak acceleration (a_max)
        limits defined in the drone parameters.

        Args:
            d: Segment distance (meters).

        Returns:
            A tuple of (T_min, T_max) in seconds.
        """
        if d <= 0:
            return (1e-3, 1.0)

        t_v = 1.5 * d / max(self.p.v_max, 1e-9)
        t_a = math.sqrt(6.0 * d / max(self.p.a_max, 1e-9))
        t_low = max(1e-3, t_v, t_a)

        # generous upper bound so optimizer can find the U-shaped minimum
        t_high = max(t_low * 4.0, self.p.t_upper_per_meter * d)
        return (t_low, t_high)

    def find_optimal_time(self, segment_vector: np.ndarray) -> SegmentResult:
        """
        Minimize segment_energy over travel time T using a bounded 1D search.

        Finds the 'sweet spot' where the sum of hover power (dominates at high T)
        and mechanical power (dominates at low T) is minimized.

        Args:
            segment_vector: 2D vector (dx, dy) of the segment.

        Returns:
            A SegmentResult containing optimized T and the resulting minimum energy E.
        """
        d = float(np.linalg.norm(segment_vector))
        if d < 0:
            return SegmentResult(distance=d, t_opt=float("nan"), e_opt=float("inf"))

        T_low, T_high = self.feasible_time_bounds(d)

        def obj(T: float) -> float:
            return self.segment_energy(segment_vector, T)

        res = minimize_scalar(obj, bounds=(T_low, T_high), method="bounded")
        return SegmentResult(distance=d, t_opt=float(res.x), e_opt=float(res.fun))

    # Build segment trajectory for plotting
    def segment_trajectory(
        self,
        A: np.ndarray,
        B: np.ndarray,
        T: float,
        steps: int | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Return a dict with arrays: t, pos (Nx2), vel (Nx2), acc (Nx2).
        """
        A = np.asarray(A, dtype=float).reshape(2)
        B = np.asarray(B, dtype=float).reshape(2)
        dvec = B - A
        d = float(np.linalg.norm(dvec))
        if d == 0:
            n = steps or 50
            t = np.linspace(0.0, T, n)
            pos = np.repeat(A[None, :], n, axis=0)
            vel = np.zeros_like(pos)
            acc = np.zeros_like(pos)
            return {"t": t, "pos": pos, "vel": vel, "acc": acc}

        u = dvec / d  # direction unit vector

        alpha = self._alpha_for_distance(d, T)
        n = steps or max(50, int(self.p.integration_steps // 4))
        t = np.linspace(0.0, T, n)
        s = self._s_profile(alpha, T, t)
        v = self._v_profile(alpha, T, t)
        a = self._a_profile(alpha, T, t)

        pos = A[None, :] + s[:, None] * u[None, :]
        vel = v[:, None] * u[None, :]
        acc = a[:, None] * u[None, :]

        return {"t": t, "pos": pos, "vel": vel, "acc": acc}
