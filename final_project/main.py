"""Orchestrates the drone routing optimization pipeline (stop at each waypoint)."""

from __future__ import annotations

import argparse

from optimizer import RoutingOptimizer
from params import (
    SimulationConfig,
    get_default_params,
    get_default_sim_config,
    get_test_sim_config,
)
from physics import DronePhysics
from plotting import Visualizer
from targets import Targets


def main(simulation_config: SimulationConfig | None = None, tsp_method: str = "brute") -> None:
    config = simulation_config or get_default_sim_config()
    params = get_default_params()

    targets = Targets(
        num_targets=config.num_targets,
        bounds=config.bounds,
        waypoint_set=config.waypoint_set,
        seed=config.seed,
    )
    waypoints = targets.generate_waypoints()
    n = waypoints.shape[0]
    distance_matrix = targets.get_distance_matrix(waypoints)

    physics = DronePhysics(params)
    optimizer = RoutingOptimizer(physics)

    energy_matrix, time_matrix = optimizer.build_energy_matrix(distance_matrix)

    method = "nearest_neighbor" if tsp_method == "nn" else "brute"
    optimal_order = optimizer.solve_tsp(energy_matrix, method=method)

    naive_order = list(range(n))

    def route_energy(order: list[int]) -> float:
        total = 0.0
        for k in range(len(order)):
            i, j = order[k], order[(k + 1) % len(order)]
            total += energy_matrix[i, j]
        return total

    def route_time(order: list[int]) -> float:
        total = 0.0
        for k in range(len(order)):
            i, j = order[k], order[(k + 1) % len(order)]
            total += time_matrix[i, j]
        return total

    optimal_energy = route_energy(optimal_order)
    naive_energy = route_energy(naive_order)

    optimal_time = route_time(optimal_order)
    naive_time = route_time(naive_order)

    print("=== Drone Routing Optimization (Stop-at-Waypoint) ===")
    print(f"Waypoints (x,y):\n{waypoints}\n")
    print(f"Naive route indices:    {naive_order}")
    print(f"Optimized route indices:{optimal_order}\n")

    print("=== Energy Comparison ===")
    print(f"Naive energy:     {naive_energy:.2f} J")
    print(f"Optimized energy: {optimal_energy:.2f} J")
    print(f"Energy saved:     {naive_energy - optimal_energy:.2f} J\n")

    print("=== Time (sum of per-segment Tmin) ===")
    print(f"Naive total time:     {naive_time:.2f} s")
    print(f"Optimized total time: {optimal_time:.2f} s")

    visualizer = Visualizer()
    visualizer.plot_energy_curve(physics, distance=1000.0)

    # Two-panel naive vs optimized route
    visualizer.plot_routes(
        waypoints=waypoints,
        naive_order=naive_order,
        optimal_order=optimal_order,
    )

    # Separate total energy figure
    visualizer.plot_total_energy(
        naive_energy=naive_energy,
        optimized_energy=optimal_energy,
        filename="total_energy.png",
    )

    print("\nPlots saved:")
    print(" - plots/energy_curve.png")
    print(" - plots/route_map.png")
    print(" - plots/total_energy.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone routing optimization (stop at each waypoint)")
    parser.add_argument("-n", "--num-targets", type=int, default=5)
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "-m",
        "--method",
        choices=["brute", "nn"],
        default="brute",
        help="TSP method: brute (exhaustive) or nn (nearest-neighbor). Brute is slow for many targets.",
    )
    args = parser.parse_args()

    cfg = get_test_sim_config() if args.test else SimulationConfig(num_targets=args.num_targets, seed=args.seed)
    main(cfg, tsp_method=args.method)