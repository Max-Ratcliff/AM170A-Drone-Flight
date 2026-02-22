"""Orchestrates the drone routing optimization pipeline."""

from config import get_default_config
from environment import MissionEnvironment
from optimizer import RoutingOptimizer
from physics import DronePhysics
from plotting import Visualizer


def main() -> None:
    """Run the full optimization pipeline: waypoints, physics, TSP, and plots."""
    config = get_default_config()
    env = MissionEnvironment(num_targets=5, bounds=(0.0, 2000.0))
    waypoints = env.generate_waypoints()
    distance_matrix = env.get_distance_matrix(waypoints)

    physics = DronePhysics(config)
    optimizer = RoutingOptimizer(physics)
    energy_matrix, optimal_velocity = optimizer.build_energy_matrix(distance_matrix)
    optimal_order = optimizer.solve_tsp(energy_matrix)

    # Total energy for the closed tour
    total_energy = 0.0
    for k in range(len(optimal_order)):
        i, j = optimal_order[k], optimal_order[(k + 1) % len(optimal_order)]
        total_energy += energy_matrix[i, j]

    print("=== Drone Routing Optimization Results ===")
    print(f"Waypoints: {waypoints}")
    print(f"Optimal route (indices): {optimal_order}")
    print(f"Optimal velocity (m/s): {optimal_velocity:.4f}")
    print(f"Total energy cost (J):  {total_energy:.2f}")

    visualizer = Visualizer()
    visualizer.plot_energy_curve(physics, distance=1000.0)
    visualizer.plot_route(waypoints, optimal_order)

    print("\nPlots saved: plots/energy_curve.png, plots/route_map.png")


if __name__ == "__main__":
    main()
