"""Orchestrates the drone routing optimization pipeline."""

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


def main(
    simulation_config: SimulationConfig | None = None,
    tsp_method: str = "brute",
) -> None:
    """Run the full optimization pipeline: waypoints, physics, TSP, and plots."""
    config = simulation_config or get_default_sim_config()
    params = get_default_params()

    targets = Targets(
        num_targets=config.num_targets,
        bounds=config.bounds,
        waypoint_set=config.waypoint_set,
        seed=config.seed,
    )
    waypoints = targets.generate_waypoints()
    num_targets = waypoints.shape[0]
    # compute the distance matrix once for all pairs of waypoints
    distance_matrix = targets.get_distance_matrix(waypoints)

    # Initialize physics and optimizer
    physics = DronePhysics(params)
    optimizer = RoutingOptimizer(physics)
    # Build energy matrix and solve TSP to find optimal route
    energy_matrix, optimal_velocity = optimizer.build_energy_matrix(distance_matrix)
    method = "nearest_neighbor" if tsp_method == "nn" else "brute"
    optimal_order = optimizer.solve_tsp(energy_matrix, method=method)

    # Naive route: sequential 0 -> 1 -> ... -> N-1 -> 0
    naive_order = list(range(num_targets))

    def route_energy(order: list[int]) -> float:
        total = 0.0
        for k in range(len(order)):
            i, j = order[k], order[(k + 1) % len(order)]
            total += energy_matrix[i, j]
        return total

    optimal_energy = route_energy(optimal_order)
    naive_energy = route_energy(naive_order)
    energy_saved = naive_energy - optimal_energy

    print("=== Drone Routing Optimization Results ===")
    print(f"Waypoints: \n{waypoints}")
    print(f"Naive route (indices): {naive_order}")
    print(f"Optimal route (indices): {optimal_order}")
    print(f"Optimal velocity (m/s): {optimal_velocity:.4f}")
    print()
    print("=== Energy Comparison ===")
    print(f"Naive route energy:   {naive_energy:.2f} J")
    print(f"Optimized route energy: {optimal_energy:.2f} J")
    print(f"Energy saved:        {energy_saved:.2f} J")

    visualizer = Visualizer()
    visualizer.plot_energy_curve(physics, distance=1000.0)
    visualizer.plot_route(
        waypoints,
        optimal_order,
        optimal_energy,
        naive_order,
        naive_energy,
    )

    print("\nPlots saved: plots/energy_curve.png, plots/route_map.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone routing optimization")
    parser.add_argument(
        "-n",
        "--num-targets",
        type=int,
        default=5,
        help="Number of waypoints (when not using --test)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible waypoints",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use fixed test waypoint set",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=["brute", "nn"],
        default="brute",
        help="TSP method: brute (exhaustive) or nn (nearest-neighbor greedy) WARNING: brute is very slow for >10 targets!",
    )
    args = parser.parse_args()

    if args.test:
        config = get_test_sim_config()
    else:
        config = SimulationConfig(
            num_targets=args.num_targets,
            seed=args.seed,
        )
    main(config, tsp_method=args.method)
