# Energy-Optimized Drone Routing Framework

A physics-grounded routing optimization framework for multi-target drone missions. Instead of minimizing geometric distance alone, it uses a quadcopter aerodynamic power model to compute true energetic cost (Joules) per segment. The framework finds optimal flight velocities to minimize aerodynamic drag and hovering power, then solves the Traveling Salesperson Problem (TSP) on the resulting energy cost matrix.

## Features

- **Aerodynamic physics engine** — Blade profile power, parasitic drag, and induced power; finds energy-minimizing velocity per segment
- **Energy cost matrix** — Converts spatial distances to Joule-based costs
- **TSP solvers** — Brute-force (optimal, N ≤ 10) and nearest-neighbor heuristic (scalable)
- **Visualization** — Route comparison plots and Joule-savings bar charts

## Installation

Python 3.10+. Dependencies: `numpy`, `scipy`, `matplotlib`.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py                              # 5 random targets, brute-force
python main.py -n 20 -m nn                  # 20 random targets, nearest-neighbor
python main.py --test                       # Fixed 6-waypoint test set brute-force
python main.py -n 8 -s 42                   # 8 random targets, seed 42 (reproducible)
```

### CLI Arguments

| Flag                  | Description                                                 |
| --------------------- | ----------------------------------------------------------- |
| `-n`, `--num-targets` | Number of random waypoints (default: 5)                     |
| `-s`, `--seed`        | Random seed for reproducible waypoints                      |
| `--test`              | Use fixed 6-waypoint test set                               |
| `-m`, `--method`      | TSP solver: `brute` (exhaustive) or `nn` (nearest-neighbor) |

## Project Structure

| File           | Purpose                                              |
| -------------- | ---------------------------------------------------- |
| `params.py`    | Drone physical constants and Simulation config       |
| `targets.py`   | 2D environment, waypoint generation, distance matrix |
| `physics.py`   | Aerodynamic power and energy equations               |
| `optimizer.py` | Velocity optimization and TSP solvers                |
| `plotting.py`  | Energy curves and route maps                         |
| `main.py`      | CLI entry point and pipeline orchestrator            |
