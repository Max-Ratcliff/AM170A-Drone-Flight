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


def main(simulation_config: SimulationConfig | None = None) -> None:
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

    # --- Three routes ---
    naive_order = list(range(n))

    # Optimized = Using brute-force or NN TSP over energy_matrix (only works for small Ns)
    optimized_order = optimizer.solve_tsp(energy_matrix, method="nearest_neighbor")
    

    # "Super" = nearest-neighbor initialization + 2-opt local improvement
    super_order = optimizer.solve_tsp(energy_matrix, method="nn_2opt")

    # Helpers functions 
    def route_energy(order: list[int]) -> float:
        total = 0.0
        for k in range(len(order)):
            i, j = order[k], order[(k + 1) % len(order)]
            total += float(energy_matrix[i, j])
        return total

    def route_time(order: list[int]) -> float:
        total = 0.0
        for k in range(len(order)):
            i, j = order[k], order[(k + 1) % len(order)]
            total += float(time_matrix[i, j])
        return total

    naive_energy = route_energy(naive_order)
    optimized_energy = route_energy(optimized_order)
    super_energy = route_energy(super_order)

    naive_time = route_time(naive_order)
    optimized_time = route_time(optimized_order)
    super_time = route_time(super_order)

    # --- Console output ---
    print("=== Drone Routing Optimization (Stop-at-Waypoint) ===")
    print(f"Waypoints (x,y):\n{waypoints}\n")

    print("=== Route Indices (cycle closes back to 0) ===")
    print(f"Naive: {naive_order}")
    print(f"Optimized: {optimized_order} (exact brute-force over energy)")
    print(f"Super: {super_order} (nearest-neighbor + 2-opt)\n")

    print("=== Total Energy Comparison ===")
    print(f"Naive energy:     {naive_energy:.2f} J")
    print(f"Optimized energy: {optimized_energy:.2f} J")
    print(f"Super energy:     {super_energy:.2f} J\n")

    print("=== Total Time (sum of per-segment Tmin) ===")
    print(f"Naive time:     {naive_time:.2f} s")
    print(f"Optimized time: {optimized_time:.2f} s")
    print(f"Super time:     {super_time:.2f} s")

    # --- Plots ---
    visualizer = Visualizer()
    visualizer.plot_energy_curve(physics, distance=1000.0)

    visualizer.plot_routes_three(
        waypoints=waypoints,
        naive_order=naive_order,
        optimized_order=optimized_order,
        super_order=super_order,
        filename="route_map.png",
    )
    visualizer.plot_total_energy_three(
        naive_energy=naive_energy,
        optimized_energy=optimized_energy,
        super_energy=super_energy,
        filename="total_energy.png",
    )

    print("\nPlots saved:")
    print(" - plots/energy_curve.png")
    print(" - plots/route_map.png")
    print(" - plots/total_energy.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Drone routing optimization (stop at each waypoint) with 3 route comparisons"
    )
    parser.add_argument("-n", "--num-targets", type=int, default=5)
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument("--test", action="store_true", help="Use a fixed waypoint set for testing")
    args = parser.parse_args()

    cfg = get_test_sim_config() if args.test else SimulationConfig(num_targets=args.num_targets, seed=args.seed)
    main(cfg)