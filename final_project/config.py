"""Physical constants for quadcopter aerodynamic modeling (DJI Phantom 4 baseline)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DroneConfig:
    """Baseline physical parameters for a standard quadcopter."""

    mass: float  # kg
    air_density: float  # kg/m^3
    rotor_area: float  # m^2
    c1: float  # Blade profile power coefficient
    c2: float  # Parasitic drag coefficient
    c3: float  # Induced power coefficient


def get_default_config() -> DroneConfig:
    """Return the default quadcopter configuration (DJI Phantom 4 baseline)."""
    return DroneConfig(
        mass=1.38,
        air_density=1.225,
        rotor_area=0.18,
        c1=100.0,
        c2=0.5,
        c3=150.0,
    )
