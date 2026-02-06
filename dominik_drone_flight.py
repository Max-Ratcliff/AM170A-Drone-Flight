# Linear Drone Flight Model with Wind Analysis - AM170A
# Dominick Rangel
#
# This code simulates a drone flying from point A to B and back,
# comparing trajectories and energy with and without 4 mph westward wind.
# Uses ODE45 (RK45) numerical integration with aerodynamic drag.

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# --- Wind and Drag Parameters ---
WIND_SPEED_MPH = 4.0
WIND_SPEED_MS = WIND_SPEED_MPH * 0.44704  # 4 mph = 1.789 m/s
V_WIND = np.array([-WIND_SPEED_MS, 0.0])  # blowing west (negative x)

RHO = 1.225  # air density (kg/m^3)
CD = 1.0  # drag coefficient for small drone
A_CROSS = 0.1  # frontal cross-sectional area (m^2)


def velocity_params(xA, yA, xB, yB, T):
    """
    Compute velocity coefficients from the analytical model.
    ax = 6(xB - xA) / T^3,  ay = 6(yB - yA) / T^3
    """
    return 6 * (xB - xA) / T**3, 6 * (yB - yA) / T**3


def analytical_position(t, ax, ay, xA, yA, T):
    """
    Analytical position: x(t) = ax*(T*t^2/2 - t^3/3) + xA
    """
    factor = T * t**2 / 2 - t**3 / 3
    return ax * factor + xA, ay * factor + yA


# =====================================================================
#  ODE SYSTEMS
# =====================================================================


def ode_no_wind(t, state, ax, ay, T):
    """
    Original ODE: no drag, no wind.
    state = [x, y, vx, vy]
    """
    _, _, vx, vy = state
    acc_x = ax * (T - 2 * t)
    acc_y = ay * (T - 2 * t)
    return [vx, vy, acc_x, acc_y]


def ode_with_wind(t, state, ax, ay, T, m):
    """
    ODE with aerodynamic drag from wind.
    Same thrust profile as no-wind.  Drag opposes motion relative to air.

    F_drag = -0.5 * rho * Cd * A * |v_rel| * v_rel
    v_rel  = v_drone - v_wind

    state = [x, y, vx, vy]
    """
    _, _, vx, vy = state

    # thrust acceleration (identical profile to no-wind case)
    acc_thrust_x = ax * (T - 2 * t)
    acc_thrust_y = ay * (T - 2 * t)

    # relative velocity of drone through air
    vrel_x = vx - V_WIND[0]
    vrel_y = vy - V_WIND[1]
    vrel_mag = np.sqrt(vrel_x**2 + vrel_y**2)

    # drag acceleration  (F_drag / m)
    k = 0.5 * RHO * CD * A_CROSS / m
    acc_drag_x = -k * vrel_mag * vrel_x
    acc_drag_y = -k * vrel_mag * vrel_y

    return [vx, vy, acc_thrust_x + acc_drag_x, acc_thrust_y + acc_drag_y]


# =====================================================================
#  SIMULATION HELPERS
# =====================================================================


def simulate_leg_no_wind(xA, yA, xB, yB, T_leg):
    """Simulate one leg without wind."""
    ax, ay = velocity_params(xA, yA, xB, yB, T_leg)
    t_eval = np.linspace(0, T_leg, 500)
    sol = solve_ivp(
        lambda t, s: ode_no_wind(t, s, ax, ay, T_leg),
        (0, T_leg),
        [xA, yA, 0.0, 0.0],
        method="RK45",
        t_eval=t_eval,
    )
    return sol, ax, ay


def simulate_leg_with_wind(xA, yA, xB, yB, T_leg, m):
    """Simulate one leg with wind drag (same thrust as no-wind)."""
    ax, ay = velocity_params(xA, yA, xB, yB, T_leg)
    t_eval = np.linspace(0, T_leg, 500)
    sol = solve_ivp(
        lambda t, s: ode_with_wind(t, s, ax, ay, T_leg, m),
        (0, T_leg),
        [xA, yA, 0.0, 0.0],
        method="RK45",
        t_eval=t_eval,
    )
    return sol, ax, ay


# =====================================================================
#  METRICS
# =====================================================================


def compute_arc_length(sol):
    """Total arc-length (distance actually traveled along path)."""
    dx = np.diff(sol.y[0])
    dy = np.diff(sol.y[1])
    return np.sum(np.sqrt(dx**2 + dy**2))


def compute_displacement(sol):
    """Straight-line displacement from start to end of a leg."""
    x0, y0 = sol.y[0][0], sol.y[1][0]
    xf, yf = sol.y[0][-1], sol.y[1][-1]
    return np.sqrt((xf - x0) ** 2 + (yf - y0) ** 2)


def compute_energy_numerical(sol, ax, ay, m, T_leg):
    """
    Motor energy = integral of |F_thrust . v| dt.
    F_thrust = m * [ax*(T-2t), ay*(T-2t)]  (same in both scenarios).
    """
    t = sol.t
    vx, vy = sol.y[2], sol.y[3]
    energy = 0.0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        t_mid = (t[i] + t[i + 1]) / 2
        Fx = m * ax * (T_leg - 2 * t_mid)
        Fy = m * ay * (T_leg - 2 * t_mid)
        vx_mid = (vx[i] + vx[i + 1]) / 2
        vy_mid = (vy[i] + vy[i + 1]) / 2
        power = abs(Fx * vx_mid + Fy * vy_mid)
        energy += power * dt
    return energy


def compute_energy_analytical(xA, yA, xB, yB, T_leg, m):
    """Analytical prediction: E = 9 m D^2 / (4 T^2) per leg."""
    D_sq = (xB - xA) ** 2 + (yB - yA) ** 2
    return 9 * m * D_sq / (4 * T_leg**2)


# =====================================================================
#  MAIN
# =====================================================================

if __name__ == "__main__":
    # --- inputs ---
    xA, yA = 0.0, 0.0
    xB, yB = 1.0, 2.0
    T_total = 1.0
    m = 1.0
    E_H = 1.0
    T_leg = T_total / 2

    # =================================================================
    #  SCENARIO 1 — No Wind  (baseline)
    # =================================================================
    sol_nw_out, ax_nw_out, ay_nw_out = simulate_leg_no_wind(xA, yA, xB, yB, T_leg)
    sol_nw_ret, ax_nw_ret, ay_nw_ret = simulate_leg_no_wind(xB, yB, xA, yA, T_leg)

    t_nw = np.concatenate([sol_nw_out.t, sol_nw_ret.t + T_leg])
    x_nw = np.concatenate([sol_nw_out.y[0], sol_nw_ret.y[0]])
    y_nw = np.concatenate([sol_nw_out.y[1], sol_nw_ret.y[1]])

    arc_nw_out = compute_arc_length(sol_nw_out)
    arc_nw_ret = compute_arc_length(sol_nw_ret)
    arc_nw_total = arc_nw_out + arc_nw_ret
    disp_nw_out = compute_displacement(sol_nw_out)

    E_nw_out = compute_energy_numerical(sol_nw_out, ax_nw_out, ay_nw_out, m, T_leg)
    E_nw_ret = compute_energy_numerical(sol_nw_ret, ax_nw_ret, ay_nw_ret, m, T_leg)
    E_nw_motion = E_nw_out + E_nw_ret
    E_nw_hover = E_H * T_total
    E_nw_total = E_nw_motion + E_nw_hover

    # analytical prediction (sanity check)
    E_nw_pred = compute_energy_analytical(
        xA, yA, xB, yB, T_leg, m
    ) + compute_energy_analytical(xB, yB, xA, yA, T_leg, m)

    # =================================================================
    #  SCENARIO 2 — 4 mph Westward Wind
    # =================================================================
    # outbound: same thrust as no-wind (A->B), but drag blows drone off
    sol_w_out, ax_w_out, ay_w_out = simulate_leg_with_wind(xA, yA, xB, yB, T_leg, m)

    # where the drone actually ends up after outbound leg
    x_end_out = sol_w_out.y[0][-1]
    y_end_out = sol_w_out.y[1][-1]

    # return: drone recalculates thrust from actual position back to A
    sol_w_ret, ax_w_ret, ay_w_ret = simulate_leg_with_wind(
        x_end_out, y_end_out, xA, yA, T_leg, m
    )

    x_final_w = sol_w_ret.y[0][-1]
    y_final_w = sol_w_ret.y[1][-1]

    t_w = np.concatenate([sol_w_out.t, sol_w_ret.t + T_leg])
    x_w = np.concatenate([sol_w_out.y[0], sol_w_ret.y[0]])
    y_w = np.concatenate([sol_w_out.y[1], sol_w_ret.y[1]])

    arc_w_out = compute_arc_length(sol_w_out)
    arc_w_ret = compute_arc_length(sol_w_ret)
    arc_w_total = arc_w_out + arc_w_ret
    disp_w_out = compute_displacement(sol_w_out)

    miss_B = np.sqrt((x_end_out - xB) ** 2 + (y_end_out - yB) ** 2)
    miss_A = np.sqrt(x_final_w**2 + y_final_w**2)

    E_w_out = compute_energy_numerical(sol_w_out, ax_w_out, ay_w_out, m, T_leg)
    E_w_ret = compute_energy_numerical(sol_w_ret, ax_w_ret, ay_w_ret, m, T_leg)
    E_w_motion = E_w_out + E_w_ret
    E_w_hover = E_H * T_total
    E_w_total = E_w_motion + E_w_hover

    # =================================================================
    #  FIGURE 1 — x(t), y(t) vs time
    # =================================================================
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(t_nw, x_nw, "b-", label="No Wind", linewidth=2)
    ax1.plot(t_w, x_w, "r-", label=f"Wind ({WIND_SPEED_MPH} mph West)", linewidth=2)
    ax1.set_xlabel("Time (s)"), ax1.set_ylabel("x (m)")
    ax1.set_title("x(t) — No Wind vs 4 mph Westward Wind")
    ax1.legend(), ax1.grid(True)

    ax2.plot(t_nw, y_nw, "b-", label="No Wind", linewidth=2)
    ax2.plot(t_w, y_w, "r-", label=f"Wind ({WIND_SPEED_MPH} mph West)", linewidth=2)
    ax2.set_xlabel("Time (s)"), ax2.set_ylabel("y (m)")
    ax2.set_title("y(t) — No Wind vs 4 mph Westward Wind")
    ax2.legend(), ax2.grid(True)

    plt.tight_layout()
    plt.savefig("trajectory_vs_time.png", dpi=150)

    # =================================================================
    #  FIGURE 2 — Parametric trajectory {x(t), y(t)}
    # =================================================================
    fig2, ax3 = plt.subplots(figsize=(10, 8))
    ax3.plot(x_nw, y_nw, "b-", label="No Wind", linewidth=2)
    ax3.plot(x_w, y_w, "r-", label=f"Wind ({WIND_SPEED_MPH} mph West)", linewidth=2)

    ax3.scatter([xA, xB], [yA, yB], color="green", s=120, zorder=5)
    ax3.annotate(
        "A (start)", (xA, yA), xytext=(10, 10), textcoords="offset points", fontsize=10
    )
    ax3.annotate(
        "B (target)", (xB, yB), xytext=(10, 10), textcoords="offset points", fontsize=10
    )

    # mark where wind-affected drone actually reached
    ax3.scatter([x_end_out], [y_end_out], color="red", marker="x", s=150, zorder=5)
    ax3.annotate(
        f"Wind outbound end\n({x_end_out:.3f}, {y_end_out:.3f})",
        (x_end_out, y_end_out),
        xytext=(15, -25),
        textcoords="offset points",
        fontsize=9,
        color="red",
    )
    ax3.scatter([x_final_w], [y_final_w], color="darkred", marker="*", s=200, zorder=5)
    ax3.annotate(
        f"Wind return end\n({x_final_w:.3f}, {y_final_w:.3f})",
        (x_final_w, y_final_w),
        xytext=(15, -25),
        textcoords="offset points",
        fontsize=9,
        color="darkred",
    )

    # wind direction arrow
    ax3.annotate(
        "",
        xy=(-0.3, 1.0),
        xytext=(0.3, 1.0),
        arrowprops=dict(arrowstyle="->", color="orange", lw=2.5),
    )
    ax3.text(
        0.0,
        1.08,
        f"Wind {WIND_SPEED_MPH} mph W",
        ha="center",
        fontsize=11,
        color="orange",
        fontweight="bold",
    )

    ax3.set_xlabel("x (m)"), ax3.set_ylabel("y (m)")
    ax3.set_title("Parametric Trajectory: No Wind vs 4 mph Westward Wind")
    ax3.legend(), ax3.grid(True), ax3.set_aspect("equal")
    plt.tight_layout()
    plt.savefig("parametric_trajectory.png", dpi=150)

    # =================================================================
    #  FIGURE 3 — Energy bar chart
    # =================================================================
    fig3, ax4 = plt.subplots(figsize=(10, 6))
    labels = [
        "Motion\n(Out)",
        "Motion\n(Return)",
        "Total\nMotion",
        "Hover",
        "Grand\nTotal",
    ]
    nw_vals = [E_nw_out, E_nw_ret, E_nw_motion, E_nw_hover, E_nw_total]
    w_vals = [E_w_out, E_w_ret, E_w_motion, E_w_hover, E_w_total]

    x_pos = np.arange(len(labels))
    width = 0.35
    bars1 = ax4.bar(
        x_pos - width / 2, nw_vals, width, label="No Wind", color="steelblue"
    )
    bars2 = ax4.bar(
        x_pos + width / 2,
        w_vals,
        width,
        label=f"Wind ({WIND_SPEED_MPH} mph W)",
        color="indianred",
    )

    ax4.set_ylabel("Energy (J)")
    ax4.set_title("Energy Comparison: No Wind vs 4 mph Westward Wind")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels)
    ax4.legend()
    ax4.grid(axis="y", alpha=0.3)
    for bar in bars1:
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar in bars2:
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig("energy_comparison.png", dpi=150)

    # =================================================================
    #  CONSOLE OUTPUT
    # =================================================================
    print("=" * 62)
    print("  WIND EFFECT ANALYSIS — 4 mph Westward Wind with Drag")
    print("=" * 62)
    print(f"\nDrag model : F_drag = -0.5 * rho * Cd * A * |v_rel| * v_rel")
    print(f"  rho = {RHO} kg/m^3,  Cd = {CD},  A = {A_CROSS} m^2")
    print(f"Wind       : {WIND_SPEED_MPH} mph West = {WIND_SPEED_MS:.4f} m/s in -x")
    print(f"Drone      : m = {m} kg,  A=({xA},{yA}) -> B=({xB},{yB}),  T = {T_total} s")

    # --- Distance ---
    print(f"\n{'─'*62}")
    print(f"  DISTANCE COMPARISON")
    print(f"{'─'*62}")
    print(f"  {'Metric':<36} {'No Wind':>10} {'With Wind':>10}")
    print(f"  {'─'*56}")
    print(f"  {'Outbound arc length (m)':<36} {arc_nw_out:>10.4f} {arc_w_out:>10.4f}")
    print(f"  {'Return arc length (m)':<36} {arc_nw_ret:>10.4f} {arc_w_ret:>10.4f}")
    print(f"  {'Total path length (m)':<36} {arc_nw_total:>10.4f} {arc_w_total:>10.4f}")
    print(
        f"  {'Path length difference (m)':<36} {'':>10} {arc_w_total - arc_nw_total:>+10.4f}"
    )
    pct_dist = (arc_w_total - arc_nw_total) / arc_nw_total * 100
    print(f"  {'Path length change (%)':<36} {'':>10} {pct_dist:>+10.2f}%")
    print(
        f"\n  {'Outbound displacement (m)':<36} {disp_nw_out:>10.4f} {disp_w_out:>10.4f}"
    )
    print(f"  {'Miss distance at B (m)':<36} {'0.0000':>10} {miss_B:>10.4f}")
    print(f"  {'Miss distance at A return (m)':<36} {'0.0000':>10} {miss_A:>10.4f}")
    print(f"\n  Wind outbound endpoint : ({x_end_out:.4f}, {y_end_out:.4f})")
    print(f"  Target B               : ({xB:.4f}, {yB:.4f})")
    print(f"  Wind return endpoint   : ({x_final_w:.4f}, {y_final_w:.4f})")
    print(f"  Target A               : ({xA:.4f}, {yA:.4f})")

    # --- Energy ---
    print(f"\n{'─'*62}")
    print(f"  ENERGY COMPARISON")
    print(f"{'─'*62}")
    print(f"  {'Metric':<36} {'No Wind':>10} {'With Wind':>10}")
    print(f"  {'─'*56}")
    print(f"  {'Outbound motion energy (J)':<36} {E_nw_out:>10.4f} {E_w_out:>10.4f}")
    print(f"  {'Return motion energy (J)':<36} {E_nw_ret:>10.4f} {E_w_ret:>10.4f}")
    print(f"  {'Total motion energy (J)':<36} {E_nw_motion:>10.4f} {E_w_motion:>10.4f}")
    print(f"  {'Hover energy (J)':<36} {E_nw_hover:>10.4f} {E_w_hover:>10.4f}")
    print(f"  {'TOTAL energy (J)':<36} {E_nw_total:>10.4f} {E_w_total:>10.4f}")
    E_diff = E_w_total - E_nw_total
    pct_energy = E_diff / E_nw_total * 100
    print(f"  {'Energy difference (J)':<36} {'':>10} {E_diff:>+10.4f}")
    print(f"  {'Energy change (%)':<36} {'':>10} {pct_energy:>+10.2f}%")

    print(f"\n  Analytical no-wind energy (check): {E_nw_pred:.4f} J")

    # --- Summary ---
    print(f"\n{'='*62}")
    print(f"  SUMMARY")
    print(f"{'='*62}")
    if E_diff > 0:
        print(f"  Wind INCREASES thrust energy by {E_diff:.4f} J  ({pct_energy:+.2f}%)")
    else:
        print(
            f"  Wind DECREASES thrust energy by {abs(E_diff):.4f} J  ({pct_energy:+.2f}%)"
        )

    if arc_w_total < arc_nw_total:
        print(
            f"  Wind REDUCES  path traveled  by {arc_nw_total - arc_w_total:.4f} m  ({pct_dist:+.2f}%)"
        )
    else:
        print(
            f"  Wind INCREASES path traveled  by {arc_w_total - arc_nw_total:.4f} m  ({pct_dist:+.2f}%)"
        )

    print(f"  Drone misses target B by {miss_B:.4f} m due to wind")
    print(f"  Drone misses return A by {miss_A:.4f} m due to wind")
    print(f"{'='*62}")
