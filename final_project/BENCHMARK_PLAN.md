# Statistical Benchmark Plan

Based on the professor's feedback, we have implemented a rigorous statistical analysis to quantify the performance of our heuristic algorithms against the global optimum.

## 1. Objective
Quantify the **optimality gap** (how far we are from the best) and the **heuristic gain** (how much 2-opt improves over basic NN) as a function of the number of waypoints ($N$).

## 2. Methodology
For each value of $N$ in the range $[3, 100]$:
1. Generate **10 random configurations** (different waypoint sets).
2. For each configuration, compute:
   - $E_{min}$: Global optimum using the **Held-Karp** solver (for $N \le 18$).
   - $E_{NN}$: Heuristic result using the **Nearest Neighbor** solver.
   - $E_{2opt}$: Improved result using **NN + 2-opt**.
3. Calculate the ratios for each trial:
   - **Optimality Gap**: $R = E_{2opt} / E_{min}$
     - This measures the precision of the 2-opt heuristic. Since finding $E_{min}$ is NP-hard, we compute this for small $N$ and extrapolate the heuristic's reliability.
   - **Heuristic Gain**: $G = E_{2opt} / E_{NN}$
     - This measures the efficiency gain of using 2-opt local search over a simple greedy approach. It quantifies the "return on investment" for the additional $O(N^2)$ computation of the swap algorithm.
4. Compute the **mean** of these ratios across the 10 trials to smooth out variance from random waypoint distributions.

## 3. Visualization Significance
The benchmark plot (`benchmark_ratios.png`) shows two critical trends:
- **Blue Line (Optimality Gap)**: Ideally sits near 1.0. An upward trend would indicate that the heuristic struggles with higher complexity. Stability near 1.0 confirms the heuristic's adequacy for real-world drone missions.
- **Green Line (Heuristic Gain)**: Sits below 1.0. A downward trend (e.g., dropping to 0.90) indicates that as the problem size $N$ grows, the greedy Nearest Neighbor approach becomes increasingly suboptimal compared to 2-opt, making 2-opt essential for larger missions.

## 4. Implementation Details
- **Step Sampling**: For $N > 20$, we sample $N$ in increments of 10 to cover a wide range ($N=100$) efficiently.
- **Reproducibility**: Each trial uses a fixed seed to ensure benchmarks can be validated.
- **Performance Constraints**: Held-Karp is capped at $N=18$ due to its $O(2^N N^2)$ time complexity, preventing the benchmark from hanging on exponential growth.

## 5. Success Metrics
- **Verification**: The 2-opt algorithm should consistently yield $G < 1.0$ (better than NN).
- **Validation**: The optimality gap should remain within a reasonable bound (e.g., $< 1.15$) for the exact solvable range, providing confidence in its performance at $N > 18$.
