import numpy as np
from optimizer import RoutingOptimizer
from params import get_default_params
from physics import DronePhysics


def validate_tsp_methods():
    print("Validating TSP methods: Brute Force vs Held-Karp")
    print("-" * 50)

    # Setup dummy physics to initialize optimizer
    params = get_default_params()
    physics = DronePhysics(params)
    optimizer = RoutingOptimizer(physics)

    # Test for different number of points
    for n in range(3, 11):
        # Generate a random cost matrix (asymmetric to be more general)
        np.random.seed(42 + n)
        cost_matrix = np.random.uniform(10, 100, size=(n, n))
        np.fill_diagonal(cost_matrix, 0)

        # Solve using Brute Force
        order_brute = optimizer.solve_tsp(cost_matrix, method="brute")
        cost_brute = optimizer._tour_cost(cost_matrix, order_brute)

        # Solve using Held-Karp
        order_hk = optimizer.solve_tsp(cost_matrix, method="held_karp")
        cost_hk = optimizer._tour_cost(cost_matrix, order_hk)

        # Validate Path Validity
        def is_valid_path(path, n):
            return len(path) == n and set(path) == set(range(n)) and path[0] == 0

        valid_brute = is_valid_path(order_brute, n)
        valid_hk = is_valid_path(order_hk, n)

        diff = abs(cost_brute - cost_hk)
        paths_match = order_brute == order_hk

        # If paths don't match, it might be due to multiple optimal solutions.
        # We still pass if costs are identical and both paths are valid.
        status = "PASSED" if diff < 1e-9 and valid_brute and valid_hk else "FAILED"

        path_status = "Exact Match" if paths_match else "Cost Match (Multi-Opt)"
        print(
            f"Nodes: {n:2d} | Brute Cost: {cost_brute:8.4f} | HK Cost: {cost_hk:8.4f} | {status} ({path_status})"
        )

        if status == "FAILED":
            print(f"  Brute Path: {order_brute}")
            print(f"  HK Path:    {order_hk}")
            return False

    print("-" * 50)
    print("All TSP validations PASSED!")
    return True


if __name__ == "__main__":
    validate_tsp_methods()
