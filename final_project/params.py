"""
Physical constants and mission configuration for drone routing.

Stop at each waypoint:
- Segment velocity profile: v(t) = alpha * t * (T - t)
- Linear drag: F = m dv/dt + C v
- Energy: integral of (hover power + |F·v|) over time
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DroneParams:
    """Physical + numerical params for the segment energy model."""
    # Physics
    mass: float # kg
    drag_coeff: float # N / (m/s)  (linear drag coefficient C)
    hover_power: float # W baseline power draw (electronics/hover)

    # Feasibility bounds (used to set lower bound on T)
    v_max: float # m/s (soft constraint)
    a_max: float # m/s^2  (soft constraint)

    # Numerical integration
    integration_steps: int = 600  # time steps for integrating energy

    # Search bounds for segment time T
    t_upper_per_meter: float = 0.7  # sec per meter (e.g., 1000m => 700s)


@dataclass
class SimulationConfig:
    """Simulation parameters: waypoints, bounds, and random seed."""
    num_targets: int = 5
    bounds: tuple[float, float] = (0.0, 2000.0)
    waypoint_set: Optional[list[tuple[float, float]]] = None
    seed: Optional[int] = None


def get_default_params() -> DroneParams:
    """Return the default quadcopter parameters (DJI Phantom 4 baseline)."""
    return DroneParams(
        mass=1.38,          # kg
        drag_coeff=1.00,    # N/(m/s)
        hover_power=60.0,  # W
        v_max=18.0,         # m/s
        a_max=6.0,          # m/s^2
        integration_steps=600,
        t_upper_per_meter=0.7,
    )


def get_default_sim_config() -> SimulationConfig:
    """Return default mission config (random waypoints)."""
    return SimulationConfig()


def get_test_sim_config() -> SimulationConfig:
    """Fixed waypoints for testing."""
    return SimulationConfig(
        num_targets=6,
        waypoint_set=[
            (0.0, 0.0),
            (500.0, 0.0),
            (500.0, 500.0),
            (0.0, 500.0),
            (250.0, 250.0),
            (750.0, 750.0),
        ],
    )