# Energy-Optimized Drone Routing Framework

A physics-grounded routing optimization framework for multi-target drone missions. Unlike standard geometric solvers that minimize distance, this framework uses a **stop-at-waypoint** aerodynamic power model to compute the true energetic cost (Joules) per segment.

## Core Methodology

### 1. Stop-to-Stop Physics Model

The framework assumes a drone must come to a complete stop at every waypoint. Each flight segment $i \to j$ follows a parabolic velocity profile:

- **Velocity:** $v(t) = \alpha t (T - t)$, where $T$ is the travel time.
- **Acceleration:** $a(t) = \alpha (T - 2t)$.
- **Distance Constraint:** $\alpha = 6d / T^3$ ensures the drone travels exactly distance $d$.

### 2. Energy Optimization

For every segment, there is an optimal travel time $T$ that minimizes total energy:
$$E(T) = \int_{0}^{T} (P_{\text{hover}} + |F(t) \cdot v(t)|) \, dt$$
where $F(t) = m \cdot a(t) + C \cdot (v(t)\hat{u} - \vec{w}) \cdot \hat{u}$ accounts for inertia, aerodynamic drag, and **wind**.

- **Short $T$:** High energy due to large acceleration and drag.
- **Long $T$:** High energy due to sustained hovering power ($P_{\text{hover}}$).
- **$T_{\text{opt}}$:** The "sweet spot" that minimizes Joules, resulting in a U-shaped energy curve.

### 3. Routing Optimization

The framework builds an **Energy Cost Matrix** using the optimized $E_{\min}$ for every possible segment and solves the Traveling Salesperson Problem (TSP) using exact or heuristic methods.

## Features

- **Aerodynamic Physics Engine:** Models mass, drag, hover power, and wind vectors.
- **Numerical Time Optimization:** Uses bounded 1D search to find $T_{\text{opt}}$ per segment.
- **Advanced TSP Solvers:**
  - **Held-Karp:** Exact dynamic programming solver (optimal for $N \le 20$).
  - **NN + 2-Opt:** Fast heuristic with local search improvement.
  - **Nearest Neighbor:** Greedy baseline.
- **Rich Visualization:** Generates publication-quality energy-vs-time curves and comparative route maps.

## Installation

Python 3.10+. Dependencies: `numpy`, `scipy`, `matplotlib`.

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Basic run with 5 targets (Runs Held-Karp, NN, and 2-Opt by default)
python main.py

# 15 targets with a 5.0 m/s headwind (x-direction)
python main.py -n 15 -w 5.0 0.0

# 50 targets using the 2-opt heuristic with clustered distribution
python main.py -n 50 -m nn_2opt -d clustered

# Overriding physical constants for a heavier drone
python main.py --mass 2.5 --drag 1.5 --hover-power 85.0
```

### CLI Arguments

| Flag                   | Default | Description                                                    |
| ---------------------- | ------- | -------------------------------------------------------------- |
| `-n`, `--num-targets`  | 5       | Number of random waypoints to generate                         |
| `-s`, `--seed`         | None    | Random seed for reproducible waypoints                         |
| `--test`               | False   | Use a fixed 6-waypoint benchmark set                           |
| `-w`, `--wind`         | 0.0 0.0 | Wind vector $(W_x, W_y)$ in m/s                                |
| `-d`, `--distribution` | uniform | Spatial layout: `uniform`, `clustered`, or `grid`              |
| `-b`, `--bounds`       | 0 2000  | Coordinate limits $(min, max)$ in meters                       |
| `-m`, `--method`       | None    | Solver: `held_karp`, `nn_2opt`, `nearest_neighbor`, or `brute` |
| `--mass`               | 1.38    | Drone mass in kg                                               |
| `--drag`               | 1.0     | Linear drag coefficient $C$                                    |
| `--hover-power`        | 60.0    | Baseline power draw in Watts                                   |

## Project Structure

| File           | Purpose                                               |
| -------------- | ----------------------------------------------------- |
| `params.py`    | Drone physical constants and simulation configuration |
| `physics.py`   | Stop-to-stop energy model and $T$ optimization logic  |
| `optimizer.py` | Energy matrix construction and TSP solver suite       |
| `targets.py`   | Waypoint generation and spatial distributions         |
| `plotting.py`  | Publication-quality visualization of results          |
| `main.py`      | Pipeline orchestration and CLI entry point            |

## Statistical Analysis & Validation

The framework includes internal plans for:

- **Numerical Convergence:** Validating integration resolution.
- **Optimality Gap:** Comparing heuristics ($E_{2opt}$) against global optima ($E_{min}$).
- **Sensitivity Analysis:** Testing how $T_{opt}$ responds to mass and drag variations.
