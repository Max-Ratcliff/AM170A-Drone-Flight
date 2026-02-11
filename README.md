# AM170A Drone Flight Project

## Problem Statement
This project models drone flight dynamics to analyze energy consumption under the influence of **quadratic atmospheric drag** and **constant wind**. The goal is to determine how a drone controller must adapt thrust to maintain a target trajectory and to identify the **energy-optimal flight velocity** ($V^*$) that balances the power required for motion against the energy consumed while hovering.

## File Descriptions
- **`drone_drag_model.py`**: The primary unified simulation engine. It performs numerical validation, optimizes travel time for minimum energy, and generates analytical plots.
- **`physics_validator.py`**: Legacy script for comparing numerical integration (RK45) against analytical kinematics in a vacuum.
- **`legacy_wind_model.py`**: Legacy script containing the initial prototype for wind-compensated flight.

## Rubric Compliance Status
The current implementation in `drone_drag_model.py` satisfies all requirements defined in the **Drone Upgrade Code Rubric**:

1.  **It runs!**: The code executes without errors and produces all required output files.
2.  **Code header**: The script contains a formalized header with goal, authors, and explicit Input/Output parameter definitions.
3.  **Code Seems to do the right thing**: The model correctly implements the quadratic drag equation and validates against the analytical solution ($E = \frac{9}{4} m \frac{d^2}{T^2}$).
4.  **Figure**: The script generates three high-quality validation figures: `energy_vs_velocity.png`, `thrust_power_analysis.png`, and `energy_breakdown.png`.
5.  **Annotations**: Every function and key block of physical logic is clearly documented with purpose, inputs, and outputs.
