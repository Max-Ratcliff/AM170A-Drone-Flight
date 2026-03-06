# Statistical Benchmark Plan

Based on the professor's feedback, we will implement a rigorous statistical analysis to quantify the performance of our heuristic algorithms against the global optimum.

## 1. Objective
Quantify the **optimality gap** (how far we are from the best) and the **heuristic gain** (how much 2-opt improves over basic NN) as a function of the number of waypoints ($N$).

## 2. Methodology
For each value of $N$ in the range $[4, 15]$:
1. Generate **10 random configurations** (different waypoint sets).
2. For each configuration, compute:
   - $E_{min}$: Global optimum using the **Held-Karp** solver.
   - $E_{NN}$: Heuristic result using the **Nearest Neighbor** solver.
   - $E_{2opt}$: Improved result using **NN + 2-opt**.
3. Calculate the ratios for each trial:
   - **Gain**: $G = E_{2opt} / E_{NN}$ (Lower is better, measures 2-opt effectiveness).
   - **Gap**: $R = E_{2opt} / E_{min}$ (Closer to 1.0 is better, measures optimality).
4. Compute the **mean** of these ratios across the 10 trials.

## 3. Implementation (Script: `benchmark.py`)
- **Loop Structure**: Outer loop for $N$, inner loop for 10 seeds.
- **Data Collection**: Store results in a dictionary or Pandas DataFrame.
- **Visualization**:
  - X-axis: Number of Waypoints ($N$).
  - Y-axis: Energy Ratio.
  - Plot two lines: Mean Gain ($E_{2opt}/E_{NN}$) and Mean Gap ($E_{2opt}/E_{min}$).
  - Font sizes must be large (Title: 22, Labels: 18, Legend: 14) for paper legibility.

## 4. Paper Restructuring
- **Validation Move**: Move the "Segment Energy vs Time" U-shaped curve analysis from *Results* to *Methods/Validation*. This establishes the physical model's correctness before showing routing results.
- **New Results**: Include the benchmark plot (`benchmark_ratios.png`) as the core statistical finding.
- **Consolidated Discussion**: Merge *Discussion* and *Conclusion* into a single, punchy section to avoid "fluff" and repetition.

## 5. Success Metrics
- A clear plot showing if the optimality gap ($E_{2opt}/E_{min}$) stays near 1.0 as $N$ increases.
- Evidence that the heuristic gain ($E_{2opt}/E_{NN}$) makes the additional computation of 2-opt worthwhile.
