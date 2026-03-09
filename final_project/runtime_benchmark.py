"""
Benchmarks the runtime of different TSP algorithms as N increases.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from optimizer import RoutingOptimizer
from params import get_default_params
from physics import DronePhysics
from plotting import save_plot
from targets import Targets

PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def benchmark_runtimes():
    params = get_default_params()
    physics = DronePhysics(params)
    optimizer = RoutingOptimizer(physics)

    # Configuration for each algorithm
    # (name, method, n_range, trials)
    configs = [
        ("Brute Force", "brute", range(3, 11), 5),
        ("Held-Karp", "held_karp", range(3, 19), 5),
        ("Nearest Neighbor", "nearest_neighbor", range(3, 101, 5), 10),
        ("NN + 2-Opt", "nn_2opt", range(3, 101, 5), 10),
    ]

    results = {}

    for name, method, n_range, trials in configs:
        print(f"Benchmarking {name}...")
        n_vals = []
        t_vals = []
        
        for n in n_range:
            times = []
            for seed in range(trials):
                targets = Targets(num_targets=n, bounds=(0, 2000), seed=seed)
                waypoints = targets.generate_waypoints()
                energy_matrix, _ = optimizer.build_energy_matrix(waypoints)

                start = time.perf_counter()
                optimizer.solve_tsp(energy_matrix, method=method)
                end = time.perf_counter()
                times.append(end - start)
            
            avg_time = np.mean(times)
            n_vals.append(n)
            t_vals.append(avg_time)
            print(f"  N={n}: {avg_time:.6f}s")
            
        results[name] = (n_vals, t_vals)

    return results


def plot_runtimes(results):
    fig = plt.figure(figsize=(10, 6))
    
    for name, (n_vals, t_vals) in results.items():
        plt.plot(n_vals, t_vals, marker='o', label=name)

    plt.yscale('log')
    plt.xlabel('Number of Waypoints (N)')
    plt.ylabel('Runtime (seconds) - Log Scale')
    plt.title('TSP Algorithm Runtime Scaling')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    save_plot(fig, "runtime_scaling.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    bench_results = benchmark_runtimes()
    plot_runtimes(bench_results)
