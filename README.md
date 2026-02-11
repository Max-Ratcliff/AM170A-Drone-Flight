# AM170A Drone Flight Project

## Problem Statement
This project models drone flight dynamics to analyze energy consumption under the influence of **quadratic atmospheric drag** and **constant wind**. The goal is to determine how a drone controller must adapt thrust to maintain a target trajectory and to identify the **energy-optimal flight velocity** ($V^*$) that balances mechanical power against hover energy.

## File Descriptions
- **`drone_drag_model.py`**: The primary unified simulation engine. It performs numerical validation, optimizes travel time for minimum energy, and generates analytical plots.
- **`physics_validator.py`**: Legacy script for comparing numerical integration (RK45) against analytical kinematics in a vacuum.
- **`legacy_wind_model.py`**: Legacy script containing the initial prototype for wind-compensated flight.

## Report Improvement Recommendations
To satisfy the **Methods for Drone Upgrade** rubric (specifically the 10pt Method and 6pt Validation sections), consider the following updates to your LaTeX document:

### 1. Update Mathematical Derivations
- **Force Balance**: Explicitly state that the drone controller uses $F_{thrust} = m a_{trajectory} - F_{drag}$. This proves the drone is "fighting" the wind to stay on the path.
- **Energy Integral**: Update your energy equation to match the script's mechanical power logic:
  $$E_{total} = \int_{0}^{T} (|F_{thrust} \cdot v_{ground}| + P_{hover}) dt$$
  This is more accurate than the previous $\|v\|^2$ approximation.

### 2. Include Pseudocode (Rubric Requirement)
The rubric requires pseudocode for "Mastery/Exceeds" in the Method section. You should outline the optimization loop:
```text
FUNCTION Calculate_Optimal_Velocity:
    FOR each velocity in range(2m/s to 25m/s):
        1. Generate trajectory (a, v) for the given speed
        2. Compute F_drag using relative velocity (v_drone - v_wind)
        3. Solve F_thrust = m*a - F_drag
        4. Integrate Power = |F_thrust * v_drone| + P_hover
    RETURN velocity that yields minimum total energy
```

### 3. Leverage New Figures for Validation
Your current report uses legacy path plots. To "convincingly demonstrate the code works," use these instead:
- **`energy_vs_velocity.png`**: This is your "Nice Figure." Use it to explain the "U-curve" (The trade-off between fighting drag at high speeds vs. wasting hover energy at low speeds).
- **`thrust_power_analysis.png`**: Use the "Wind Assist" (green) vs. "Extra Cost" (red) areas to explain how the drone saves energy when moving with the wind.

### 4. Strengthen the Validation Discussion
In the "Validation" section of the report, mention that you verified the numerical integrator by running a zero-drag case. State that the numerical result matched the analytical solution ($E = \frac{9}{4} m \frac{d^2}{T^2}$) with an error of $\approx 10^{-12}$, proving the solver's precision.
