# Additional Validation Strategies

To strengthen the "Model & Methods" section of the paper, we can implement the following validation tests. These ensure the physics engine is reliable and the optimization is meaningful.

## Analytical Baseline Comparison (The "Sanity Check")

**Goal:** Compare the hybrid model against a simplified case where we know the answer.

- **Method:** Create a "Vacuum Model" where drag $C = 0$. In this case, the energy integral $\int (P_{hover} + |ma \cdot v|) dt$ can be solved analytically.
- **Validation:** Compare the code's output for $C=0$ against the hand-calculated analytical value. Matching these confirms the integration logic is sound.

## Physical Sensitivity Analysis

**Goal:** Verify that the "Sweet Spot" travel time ($T_{opt}$) responds correctly to physics.

- **Method:** Plot multiple energy curves on one graph while varying one parameter at a time:
  - **Mass ($m$):** Increasing mass should shift $T_{opt}$ higher (slower flight to save power).
  - **Drag ($C$):** Increasing drag should shift $T_{opt}$ higher (slower flight to reduce aerodynamic cost).
  - **Hover Power ($P_{hover}$):** Increasing hover power should shift $T_{opt}$ lower (faster flight to minimize hover time).
- **Validation:** If the "U-shape" shifts in the expected directions, it confirms the trade-offs are modeled correctly.

## Wind Symmetry Test

**Goal:** Validate the vector math in the wind model.

- **Method:** Compare the energy for a 1000m segment in three cases:
  1. No wind.
  2. 5 m/s headwind.
  3. 5 m/s tailwind.
- **Validation:** Headwind should significantly increase energy; tailwind should decrease energy (up to a point). This confirms the $F = ma + C(v - w)$ vector implementation is correct.

## "Distance-Only" Divergence Test

**Goal:** Validate that Energy-TSP is actually different from Distance-TSP.

- **Method:** Solve a 10-target mission using standard Euclidean distance, then calculate how much energy that route consumes using our physics model. Compare it to our Energy-optimized route.
- **Validation:** Identify specific configurations where the "Shortest Path" is >5% more expensive than the "Energy Path" (e.g., due to wind or frequent stops). This justifies the existence of the research.
