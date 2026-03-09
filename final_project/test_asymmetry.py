import numpy as np
from params import get_default_params
from physics import DronePhysics

def test_wind_asymmetry():
    params = get_default_params(mass=1.38, drag_coeff=1.0, hover_power=60.0)
    # Wind 15m/s in +X direction
    physics = DronePhysics(params, wind_vector=(15.0, 0.0))
    
    A = np.array([0.0, 0.0])
    B = np.array([1000.0, 0.0])
    
    # Segment A -> B (With wind)
    res_ab = physics.find_optimal_time(B - A)
    # Segment B -> A (Against wind)
    res_ba = physics.find_optimal_time(A - B)
    
    print(f"Segment 1000m with 15m/s tail/headwind:")
    print(f"  A -> B (Tailwind): E = {res_ab.e_opt:.2f} J, T = {res_ab.t_opt:.1f}s")
    print(f"  B -> A (Headwind): E = {res_ba.e_opt:.2f} J, T = {res_ba.t_opt:.1f}s")
    
    ratio = res_ba.e_opt / res_ab.e_opt
    print(f"  Asymmetry Ratio (Against/With): {ratio:.2f}")

if __name__ == "__main__":
    test_wind_asymmetry()
