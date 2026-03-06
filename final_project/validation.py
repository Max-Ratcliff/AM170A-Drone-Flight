"""
Validation scripts for the drone physics and energy-optimized routing.
Includes analytical comparisons and divergence tests.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from params import get_default_params
from physics import DronePhysics
from targets import Targets
from optimizer import RoutingOptimizer

PLOTS_DIR = Path(__file__).resolve().parent / "plots"

def vacuum_sanity_check():
    """
    Validates numerical integration against an analytical solution for a zero-drag case.
    E = P_hover * T + mass * integral(|a * v|) dt
    For stop-to-stop parabolic: integral(|a * v|) dt = (1.5 * d / T)^2 * mass (approx)
    Actually, let's derive it more precisely:
    v(t) = alpha * t * (T-t), a(t) = alpha * (T - 2t)
    Work = ∫ m |a(t) v(t)| dt
    Since v(t) is always positive, and a(t) flips sign at T/2:
    Work = m * alpha^2 * [ ∫_0^{T/2} (T-2t)t(T-t) dt - ∫_{T/2}^T (T-2t)t(T-t) dt ]
    Evaluating the integral: 
    ∫ (T^2 t - 3Tt^2 + 2t^3) dt = T^2 t^2/2 - Tt^3 + t^4/2
    At T/2: T^2 (T^2/8) - T (T^3/8) + (T^4/32) = T^4/64
    Full integral = 2 * (T^4/64) = T^4/32
    Work = m * alpha^2 * T^4 / 32
    Since alpha = 6d / T^3:
    Work = m * (36 d^2 / T^6) * (T^4 / 32) = (36/32) * m * d^2 / T^2 = 1.125 * m * d^2 / T^2
    """
    print("Running Vacuum Sanity Check (C=0)...")
    mass = 1.38
    d = 1000.0
    T = 100.0
    hover_power = 60.0
    
    # Numerical
    params = get_default_params(mass=mass, drag_coeff=0.0, hover_power=hover_power)
    physics = DronePhysics(params)
    e_num = physics.segment_energy(np.array([d, 0]), T)
    
    # Analytical
    # Work = ∫ |m * a * v| dt = 2 * (1/2 * m * v_max^2) = m * v_max^2
    # v_max = 1.5 * d / T
    # Work = m * (1.5 * d / T)^2 = 2.25 * m * d^2 / T^2
    work_analytical = 2.25 * mass * (d**2) / (T**2)
    e_analytical = hover_power * T + work_analytical
    
    diff = abs(e_num - e_analytical) / e_analytical
    print(f"  Numerical Energy: {e_num:.4f} J")
    print(f"  Analytical Energy: {e_analytical:.4f} J")
    print(f"  Relative Difference: {diff:.6%}")
    
    if diff < 0.001:
        print("  SUCCESS: Numerical integration matches analytical model within 0.1%")
    else:
        print("  WARNING: Significant divergence detected!")

def distance_divergence_test():
    """
    Compares the Shortest Distance route vs the Energy Optimized route.
    """
    print("\nRunning Distance vs Energy Divergence Test...")
    # Using a 10-target clustered set where geometry might be tricky
    n = 15
    seed = 44
    
    # Heavy drone with high drag to make movement expensive
    params = get_default_params(mass=4.0, drag_coeff=3.0, hover_power=40.0)
    physics = DronePhysics(params, wind_vector=(15.0, 5.0))
    optimizer = RoutingOptimizer(physics)
    
    targets = Targets(num_targets=n, bounds=(0, 2000), seed=seed, distribution="uniform")
    waypoints = targets.generate_waypoints()
    
    # Energy Matrix
    energy_matrix, _ = optimizer.build_energy_matrix(waypoints)
    
    # Distance Matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(waypoints[i] - waypoints[j])
            
    # Solve for Minimum Distance (using HK on distance matrix)
    dist_order = optimizer.solve_tsp(dist_matrix, method="held_karp")
    e_dist_route = optimizer._tour_cost(energy_matrix, dist_order)
    d_dist_route = optimizer._tour_cost(dist_matrix, dist_order)
    
    # Solve for Minimum Energy (using HK on energy matrix)
    energy_order = optimizer.solve_tsp(energy_matrix, method="held_karp")
    e_energy_route = optimizer._tour_cost(energy_matrix, energy_order)
    d_energy_route = optimizer._tour_cost(dist_matrix, energy_order)
    
    savings = (e_dist_route - e_energy_route) / e_dist_route
    print(f"  Shortest Path Energy: {e_dist_route:.2f} J (Distance: {d_dist_route:.1f}m)")
    print(f"  Optimal Energy Path:  {e_energy_route:.2f} J (Distance: {d_energy_route:.1f}m)")
    print(f"  Energy Savings: {savings:.2%}")
    
    if savings > 0.05:
        print(f"  SUCCESS: Identified >5% energy savings over shortest path.")
    else:
        print(f"  NOTE: Energy path is similar to shortest path for this configuration.")

if __name__ == "__main__":
    vacuum_sanity_check()
    distance_divergence_test()
