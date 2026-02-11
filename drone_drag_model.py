"""
AM170A: Unified Drone Flight Model with Wind and Drag Compensation
Authors: Dominick Rangel, Max Ratcliff, Edmund Xu, Alex Brezny

This script provides a complete simulation environment for drone flight,
incorporating quadratic atmospheric drag and constant wind.

Input Parameters:
- Start and End Coordinates: (xA, yA) and (xB, yB)
- Total Mission Time: T_total (seconds)
- Drone Mass: MASS (kg)
- Hover Power: E_HOVER (Watts)

Outputs:
1. Validation of the numerical model against the analytical solution for the zero-drag case.
2. Optimal total mission time (T*) that minimizes energy consumption with drag and wind.
3. Visualizations:
    - Thrust and Power Profiles: Comparing no-wind and with-wind scenarios.
    - Energy vs. Velocity Plot: Showing the "Upgrade" effect of drag and wind on optimal speed.
    - Energy Breakdown Bar Chart: Comparing motion and hover energy components for both models.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

# --- Physics & Environment Constants ---
RHO = 1.225  # Air density (kg/m^3)
CD = 1.0  # Drag coefficient
A_CROSS = 0.1  # Frontal area (m^2)
WIND_SPEED_MPH = 4.0
WIND_SPEED_MS = WIND_SPEED_MPH * 0.44704
V_WIND = np.array([-WIND_SPEED_MS, 0.0])  # Constant westward wind

# --- Drone Parameters ---
MASS = 1.5  # kg
E_HOVER = 50.0  # Watts (Hovering power)

# =====================================================================
#  TRAJECTORY GENERATION
# =====================================================================


def get_trajectory_profiles(xA, yA, xB, yB, T_leg, N=2000):
    """
    Generates time-series for position, velocity, and acceleration.
    Uses a jerk-minimizing profile: v(t) is a downward parabola.
    """
    ax_const = 6 * (xB - xA) / T_leg**3
    ay_const = 6 * (yB - yA) / T_leg**3

    t = np.linspace(0, T_leg, N)

    # Analytical Kinematics
    vx = ax_const * t * (T_leg - t)
    vy = ay_const * t * (T_leg - t)
    acx = ax_const * (T_leg - 2 * t)
    acy = ay_const * (T_leg - 2 * t)

    # Position (for validation plotting)
    x = ax_const * (T_leg * t**2 / 2 - t**3 / 3) + xA
    y = ay_const * (T_leg * t**2 / 2 - t**3 / 3) + yA

    return t, x, y, vx, vy, acx, acy


# =====================================================================
#  FORCE AND ENERGY CALCULATIONS
# =====================================================================


def calculate_thrust(vx, vy, acx, acy, use_drag=True):
    """
    F_thrust = m*a - F_drag
    F_drag = -0.5 * rho * Cd * A * |v_rel| * v_rel
    """
    if not use_drag:
        return MASS * acx, MASS * acy

    vrel_x = vx - V_WIND[0]
    vrel_y = vy - V_WIND[1]
    vrel_mag = np.sqrt(vrel_x**2 + vrel_y**2)

    k = 0.5 * RHO * CD * A_CROSS
    Fx = MASS * acx + k * vrel_mag * vrel_x
    Fy = MASS * acy + k * vrel_mag * vrel_y
    return Fx, Fy


def compute_leg_energy(xA, yA, xB, yB, T_leg, use_drag=True):
    """Integrates mechanical power and adds hover energy."""
    t, _, _, vx, vy, acx, acy = get_trajectory_profiles(xA, yA, xB, yB, T_leg)
    Fx, Fy = calculate_thrust(vx, vy, acx, acy, use_drag)

    # Power = |F_thrust . v_drone|
    power_mech = np.abs(Fx * vx + Fy * vy)

    e_motion = np.trapz(power_mech, t)
    e_hover = E_HOVER * T_leg
    return e_motion, e_hover


def roundtrip_total_energy(xA, yA, xB, yB, T_total, use_drag=True):
    """Calculates total energy for Outbound and Return."""
    T_leg = T_total / 2.0
    em_out, eh_out = compute_leg_energy(xA, yA, xB, yB, T_leg, use_drag)
    em_ret, eh_ret = compute_leg_energy(xB, yB, xA, yA, T_leg, use_drag)
    return (em_out + em_ret) + (eh_out + eh_ret)


# =====================================================================
#  3. PLOTTING FUNCTIONS
# =====================================================================


def generate_visuals(xA, yA, xB, yB):
    T_nominal = 30.0
    T_leg = T_nominal / 2
    dist = np.sqrt((xB - xA) ** 2 + (yB - yA) ** 2)

    # --- Plot 1: Thrust & Power Profiles ---
    t, _, _, vx, vy, acx, acy = get_trajectory_profiles(xA, yA, xB, yB, T_leg)
    Fx_nw, Fy_nw = calculate_thrust(vx, vy, acx, acy, False)
    Fx_w, Fy_w = calculate_thrust(vx, vy, acx, acy, True)

    P_nw = np.abs(Fx_nw * vx + Fy_nw * vy)
    P_w = np.abs(Fx_w * vx + Fy_w * vy)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t, np.sqrt(Fx_nw**2 + Fy_nw**2), "b--", label="No Wind")
    plt.plot(t, np.sqrt(Fx_w**2 + Fy_w**2), "r-", label="With Wind")
    plt.title("Thrust Magnitude (Outbound)")
    plt.xlabel("Time (s)"), plt.ylabel("Force (N)"), plt.legend(), plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(t, P_nw, "b--", label="No Wind")
    plt.plot(t, P_w, "r-", label="With Wind")
    plt.fill_between(
        t, P_nw, P_w, where=(P_w > P_nw), color="red", alpha=0.2, label="Extra Cost"
    )
    plt.fill_between(
        t, P_nw, P_w, where=(P_w < P_nw), color="green", alpha=0.2, label="Wind Assist"
    )
    plt.title("Mechanical Power (Outbound)")
    plt.xlabel("Time (s)"), plt.ylabel("Power (W)"), plt.legend(), plt.grid(True)
    plt.tight_layout()
    plt.savefig("thrust_power_profiles.png")

    # --- Plot 2: Energy vs. Velocity (The "Upgrade" Validation) ---
    v_range = np.linspace(2, 25, 100)  # Average Velocity in m/s
    T_sweep = (2 * dist) / v_range

    E_nw = [roundtrip_total_energy(xA, yA, xB, yB, T, False) for T in T_sweep]
    E_w = [roundtrip_total_energy(xA, yA, xB, yB, T, True) for T in T_sweep]

    opt_nw = minimize_scalar(
        lambda T: roundtrip_total_energy(xA, yA, xB, yB, T, False),
        bounds=(5, 100),
        method="bounded",
    )
    opt_w = minimize_scalar(
        lambda T: roundtrip_total_energy(xA, yA, xB, yB, T, True),
        bounds=(5, 100),
        method="bounded",
    )

    v_opt_nw = (2 * dist) / opt_nw.x
    v_opt_w = (2 * dist) / opt_w.x

    plt.figure(figsize=(10, 6))
    plt.plot(v_range, E_nw, "b-", label="Ideal Model (No Drag)", lw=2)
    plt.plot(
        v_range, E_w, "r-", label=f"Upgraded Model ({WIND_SPEED_MPH}mph Wind)", lw=2
    )
    plt.scatter(v_opt_nw, opt_nw.fun, color="blue", s=100, edgecolors="k", zorder=5)
    plt.scatter(v_opt_w, opt_w.fun, color="red", s=100, edgecolors="k", zorder=5)

    plt.title("Mission Energy vs. Flight Velocity", fontsize=14)
    plt.xlabel("Average Ground Velocity (m/s)", fontsize=12)
    plt.ylabel("Total Round-Trip Energy (J)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Annotate the optimal speeds
    plt.annotate(
        f"V* = {v_opt_nw:.1f} m/s",
        (v_opt_nw, opt_nw.fun),
        xytext=(10, 20),
        textcoords="offset points",
        color="blue",
    )
    plt.annotate(
        f"V* = {v_opt_w:.1f} m/s",
        (v_opt_w, opt_w.fun),
        xytext=(10, -30),
        textcoords="offset points",
        color="red",
    )

    plt.savefig("energy_vs_velocity.png")

    # --- Plot 3: Energy Comparison Bar Chart ---
    em_nw_o, eh_nw_o = compute_leg_energy(xA, yA, xB, yB, T_leg, False)
    em_nw_r, eh_nw_r = compute_leg_energy(xB, yB, xA, yA, T_leg, False)
    em_w_o, eh_w_o = compute_leg_energy(xA, yA, xB, yB, T_leg, True)
    em_w_r, eh_w_r = compute_leg_energy(xB, yB, xA, yA, T_leg, True)

    labels = ["Motion (Out)", "Motion (Ret)", "Hover"]
    nw_vals = [em_nw_o, em_nw_r, eh_nw_o + eh_nw_r]
    w_vals = [em_w_o, em_w_r, eh_w_o + eh_w_r]

    x = np.arange(len(labels))
    plt.figure(figsize=(8, 6))
    plt.bar(x - 0.2, nw_vals, 0.4, label="No Wind", color="steelblue")
    plt.bar(x + 0.2, w_vals, 0.4, label="With Wind", color="indianred")
    plt.xticks(x, labels)
    plt.ylabel("Energy (J)"), plt.title(
        f"Energy Breakdown (T={T_nominal}s)"
    ), plt.legend(), plt.grid(axis="y")
    plt.savefig("energy_comparison.png")


# =====================================================================
#  MAIN EXECUTION
# =====================================================================


def main():
    print("=" * 60)
    print("      DRONE DRAG MODEL: UNIFIED SIMULATION & VALIDATION")
    print("=" * 60)

    # Config
    xA, yA, xB, yB = 0, 0, 100, 50
    dist = np.sqrt((xB - xA) ** 2 + (yB - yA) ** 2)
    T_val = 20.0
    em_num, _ = compute_leg_energy(xA, yA, xB, yB, T_val / 2, use_drag=False)
    em_ana = (9.0 / 4.0) * MASS * (dist**2 / (T_val / 2) ** 2)
    err = abs(em_num - em_ana)

    print(f"\n[1/3] VALIDATION (Zero-Drag Case, T_leg={T_val/2}s)")
    print(f"      Numerical Motion Energy:  {em_num:.4f} J")
    print(f"      Analytical Motion Energy: {em_ana:.4f} J")
    print(f"      Absolute Error:           {err:.2e} J")

    # Optimization
    print(f"\n[2/3] OPTIMIZATION (Finding T*)")
    opt_w = minimize_scalar(
        lambda T: roundtrip_total_energy(xA, yA, xB, yB, T, True),
        bounds=(5, 100),
        method="bounded",
    )
    print(f"      Optimal Time (T*): {opt_w.x:.2f} s")
    print(f"      Minimum Energy:    {opt_w.fun:.2f} J")

    # Visuals
    print(f"\n[3/3] GENERATING VISUALS...")
    generate_visuals(xA, yA, xB, yB)
    print("      - thrust_power_profiles.png")
    print("      - drone_upgrade_validation.png")
    print("      - energy_comparison.png")

    print("\n[SUCCESS] Simulation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
