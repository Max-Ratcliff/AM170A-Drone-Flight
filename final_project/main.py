"""Orchestrates the drone routing optimization pipeline with stop-at-waypoint physics."""

import argparse
from typing import Dict, List

import numpy as np
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


def main(
    simulation_config: SimulationConfig | None = None,
    overrides: dict | None = None,
    tsp_method: str | None = None,
) -> None:
    """Run the full optimization pipeline: waypoints, physics, TSP, and plots."""
    config = simulation_config or get_default_sim_config()

    # Extract overrides for physical params
    ov = overrides or {}
    params = get_default_params(
        mass=ov.get("mass", 1.38),
        drag_coeff=ov.get("drag", 1.00),
        hover_power=ov.get("hover_power", 60.0),
    )

    targets = Targets(
        num_targets=config.num_targets,
        bounds=config.bounds,
        waypoint_set=config.waypoint_set,
        distribution=config.distribution,
        seed=config.seed,
    )
    waypoints = targets.generate_waypoints()
    num_targets = waypoints.shape[0]

    # Initialize physics and optimizer with wind
    physics = DronePhysics(params, wind_vector=config.wind_vector)
    optimizer = RoutingOptimizer(physics)

    # Build energy matrix
    energy_matrix, _ = optimizer.build_energy_matrix(waypoints)

    # Define route cost helper
    def route_energy(order: List[int]) -> float:
        total = 0.0
        for k in range(len(order)):
            i, j = order[k], order[(k + 1) % len(order)]
            total += energy_matrix[i, j]
        return total

    # Dictionary to store results for all computed solvers
    orders: Dict[str, List[int]] = {}
    results_energy: Dict[str, float] = {}

    # Always compute Naive
    naive_order = list(range(num_targets))
    orders["Naive"] = naive_order
    results_energy["Naive"] = route_energy(naive_order)

    if tsp_method:
        # User specified a single method
        methods_to_run = [tsp_method]
        primary_method = tsp_method
    else:
        # Default: run all safe solvers (skip brute-force automatically)
        methods_to_run = ["nearest_neighbor", "nn_2opt", "held_karp"]
        primary_method = "held_karp"

    for m in methods_to_run:
        order = optimizer.solve_tsp(energy_matrix, method=m)
        orders[m] = order
        results_energy[m] = route_energy(order)

    # Primary results for console output
    optimal_energy = results_energy[primary_method]
    naive_energy = results_energy["Naive"]
    energy_saved = naive_energy - optimal_energy

    print("=== Drone Routing Optimization Results (Stop-at-Waypoint) ===")
    print(f"Waypoints: \n{waypoints}")
    print(f"Wind Vector (m/s): {config.wind_vector}")
    print(
        f"Physics: mass={params.mass}kg, drag={params.drag_coeff}, hover={params.hover_power}W"
    )
    print(f"Naive route energy: {naive_energy:.2f} J")
    print(f"Optimal ({primary_method}) energy: {optimal_energy:.2f} J")
    print(f"Energy saved: {energy_saved:.2f} J")

    visualizer = Visualizer()

    # Energy diagnostic curve (Multi-curve sensitivity analysis)
    visualizer.plot_energy_curve(physics, np.array([1000.0, 0.0]))

    # Comprehensive Route Grid (2x2 comparison including Wind)
    visualizer.plot_route_grid(
        waypoints,
        orders,
        results_energy,
        wind_vector=physics.wind,
        filename="route_comparison.png",
    )

    # Bar chart of all computed solvers
    visualizer.plot_energy_comparison(results_energy, filename="total_energy.png")

    print("\nPlots saved to plots/:")
    print(" - energy_curve.png (Optimal Energy: No-Drag vs With-Drag)")
    print(" - route_comparison.png (Grid comparison with Wind)")
    print(" - total_energy.png (Solver performance bar chart)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone routing optimization")

    # Simulation Config
    parser.add_argument(
        "-n", "--num-targets", type=int, default=5, help="Number of waypoints"
    )
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument("--test", action="store_true", help="Use fixed test set")
    parser.add_argument(
        "-w",
        "--wind",
        type=float,
        nargs=2,
        default=[0.0, 0.0],
        metavar=("WX", "WY"),
        help="Wind vector (m/s)",
    )
    parser.add_argument(
        "-d",
        "--distribution",
        choices=["uniform", "clustered", "grid"],
        default="uniform",
        help="Spatial distribution",
    )
    parser.add_argument(
        "-b",
        "--bounds",
        type=float,
        nargs=2,
        default=[0.0, 2000.0],
        metavar=("MIN", "MAX"),
        help="Coordinate bounds (m)",
    )

    # Physical Constants
    parser.add_argument("--mass", type=float, default=1.38, help="Drone mass (kg)")
    parser.add_argument(
        "--drag", type=float, default=1.0, help="Linear drag coefficient C"
    )
    parser.add_argument(
        "--hover-power", type=float, default=60.0, help="Baseline hover power (W)"
    )

    # Optimizer
    parser.add_argument(
        "-m",
        "--method",
        choices=["brute", "nearest_neighbor", "held_karp", "nn_2opt"],
        default=None,
        help="TSP method. If omitted, runs NN, 2-opt, and Held-Karp.",
    )

    args = parser.parse_args()

    # Physical overrides dictionary
    overrides = {"mass": args.mass, "drag": args.drag, "hover_power": args.hover_power}

    if args.test:
        config = get_test_sim_config()
    else:
        config = SimulationConfig(
            num_targets=args.num_targets,
            seed=args.seed,
            wind_vector=tuple(args.wind),
            distribution=args.distribution,
            bounds=tuple(args.bounds),
        )

    main(config, overrides=overrides, tsp_method=args.method)
