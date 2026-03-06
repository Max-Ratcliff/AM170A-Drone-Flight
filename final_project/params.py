"""
Physical constants and mission configuration for drone routing.
"""

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
    wind_vector: tuple[float, float] = (0.0, 0.0)
    distribution: str = "uniform"


def get_default_params(
    mass: float = 1.38,
    drag_coeff: float = 1.00,
    hover_power: float = 60.0,
    v_max: float = 18.0,
    a_max: float = 6.0,
) -> DroneParams:
    """Return the quadcopter parameters with optional overrides."""
    return DroneParams(
        mass=mass,
        drag_coeff=drag_coeff,
        hover_power=hover_power,
        v_max=v_max,
        a_max=a_max,
        integration_steps=600,
        t_upper_per_meter=0.7,
    )


def get_default_sim_config() -> SimulationConfig:
    """Standard 5-target random mission."""
    return SimulationConfig(num_targets=5)


def get_test_sim_config() -> SimulationConfig:
    """Fixed waypoints for testing."""
    return SimulationConfig(
        num_targets=6,
        waypoint_set=[
            (0.0, 0.0),
            (1000.0, 0.0),
            (1000.0, 1000.0),
            (0.0, 1000.0),
            (250.0, 250.0),
            (750.0, 750.0),
        ],
        wind_vector=(5.0, -2.0),
    )
