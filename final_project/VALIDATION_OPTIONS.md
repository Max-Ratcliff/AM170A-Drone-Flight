# Additional Validation Strategies

To strengthen the "Model & Methods" section of the paper, we can implement the following validation tests. These ensure the physics engine is reliable and the optimization is meaningful.

## Analytical Baseline Comparison (The "Sanity Check")

**Goal:** Compare the hybrid model against a simplified case where we know the answer.

- **Method:** Create a "Vacuum Model" where drag $C = 0$. In this case, the energy integral $\int (P_{hover} + |ma \cdot v|) dt$ can be solved analytically.
- **Validation:** Compare the code's output for $C=0$ against the hand-calculated analytical value. Matching these confirms the integration logic is sound.

## "Distance-Only" Divergence Test

**Goal:** Validate that Energy-TSP is actually different from Distance-TSP.

- **Method:** Solve a 10-target mission using standard Euclidean distance, then calculate how much energy that route consumes using our physics model. Compare it to our Energy-optimized route.
- **Validation:** Identify specific configurations where the "Shortest Path" is >5% more expensive than the "Energy Path" (e.g., due to wind or frequent stops). This justifies the existence of the research.
