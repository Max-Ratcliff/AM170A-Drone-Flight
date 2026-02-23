"""
Physical constants and mission configuration for drone routing.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DroneParams:
    """Baseline physical parameters for a standard quadcopter."""

    mass: float  # kg
    air_density: float  # kg/m^3
    rotor_area: float  # m^2
    c1: float  # Blade profile power coefficient
    c2: float  # Parasitic drag coefficient
    c3: float  # Induced power coefficient


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
        mass=1.38,
        air_density=1.225,
        rotor_area=0.18,
        c1=100.0,
        c2=0.5,
        c3=150.0,
    )


def get_default_sim_config() -> SimulationConfig:
    """Return default mission config (random waypoints)."""
    return SimulationConfig()


def get_test_sim_config() -> SimulationConfig:
    """Return a fixed waypoint set for reproducible testing."""
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
