"""Orchestrates the drone routing optimization pipeline."""

from optimizer import RoutingOptimizer
from params import get_default_params
from physics import DronePhysics
from plotting import Visualizer
from targets import Targets


def main() -> None:
    """Run the full optimization pipeline: waypoints, physics, TSP, and plots."""
    num_targets = 5
    # params can be pulled from params.py or set here
    params = get_default_params()
    targets = Targets(num_targets=num_targets, bounds=(0.0, 2000.0))
    waypoints = targets.generate_waypoints()
    # compute the distance matrix once for all pairs of waypoints
    distance_matrix = targets.get_distance_matrix(waypoints)

    # Initialize physics and optimizer
    physics = DronePhysics(params)
    optimizer = RoutingOptimizer(physics)
    # Build energy matrix and solve TSP to find optimal route
    energy_matrix, optimal_velocity = optimizer.build_energy_matrix(distance_matrix)
    optimal_order = optimizer.solve_tsp(energy_matrix, method="brute")

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
    main()
