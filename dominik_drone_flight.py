# Linear Drone Flight Model with Wind Compensation - AM170A
# Dominick Rangel
#
# Drone adapts its thrust to maintain the ideal no-wind trajectory
# despite 4 mph westward wind drag.  Infinite energy assumed.
# Compares energy cost and optimal travel time with/without wind.

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


# --- Wind and Drag Parameters ---
WIND_SPEED_MPH = 4.0
WIND_SPEED_MS = WIND_SPEED_MPH * 0.44704   # 4 mph → 1.789 m/s
V_WIND = np.array([-WIND_SPEED_MS, 0.0])   # blowing west (negative x)

RHO = 1.225       # air density (kg/m^3)
CD = 1.0          # drag coefficient for small drone
A_CROSS = 0.1     # frontal cross-sectional area (m^2)


# =====================================================================
#  ANALYTICAL TRAJECTORY  (same for both scenarios)
# =====================================================================

def velocity_params(xA, yA, xB, yB, T):
    """ax = 6(xB-xA)/T^3,  ay = 6(yB-yA)/T^3"""
    return 6 * (xB - xA) / T**3, 6 * (yB - yA) / T**3


def trajectory_profiles(xA, yA, xB, yB, T_leg, N=2000):
    """
    Return time array and analytical v(t), a(t) for one leg.
    Position : x(t) = ax*(T*t^2/2 - t^3/3) + xA
    Velocity : vx(t) = ax * t * (T - t)
    Accel    : ax_acc(t) = ax * (T - 2t)
    """
    ax, ay = velocity_params(xA, yA, xB, yB, T_leg)
    t = np.linspace(0, T_leg, N)
    vx = ax * t * (T_leg - t)
    vy = ay * t * (T_leg - t)
    acx = ax * (T_leg - 2 * t)
    acy = ay * (T_leg - 2 * t)
    return t, vx, vy, acx, acy


# =====================================================================
#  THRUST AND ENERGY — NO WIND
# =====================================================================

def thrust_no_wind(acx, acy, m):
    """F_thrust = m * a_analytical."""
    return m * acx, m * acy


def leg_energy_no_wind(xA, yA, xB, yB, T_leg, m, N=2000):
    """E = integral |F_thrust . v| dt  (no wind)."""
    t, vx, vy, acx, acy = trajectory_profiles(xA, yA, xB, yB, T_leg, N)
    Fx, Fy = thrust_no_wind(acx, acy, m)
    power = np.abs(Fx * vx + Fy * vy)
    return np.trapz(power, t)


# =====================================================================
#  THRUST AND ENERGY — WITH WIND (compensated)
# =====================================================================

def thrust_with_wind(vx, vy, acx, acy, m):
    """
    Drone maintains the same trajectory despite wind drag.
    m * a_analytical = F_thrust + F_drag
    F_thrust = m * a_analytical - F_drag
             = m * a + 0.5 * rho * Cd * A * |v_rel| * v_rel
    (drag opposes relative motion, so compensating adds it back)
    """
    vrel_x = vx - V_WIND[0]
    vrel_y = vy - V_WIND[1]
    vrel_mag = np.sqrt(vrel_x**2 + vrel_y**2)

    Fx = m * acx + 0.5 * RHO * CD * A_CROSS * vrel_mag * vrel_x
    Fy = m * acy + 0.5 * RHO * CD * A_CROSS * vrel_mag * vrel_y
    return Fx, Fy


def leg_energy_wind(xA, yA, xB, yB, T_leg, m, N=2000):
    """E = integral |F_thrust_wind . v| dt  (wind compensated)."""
    t, vx, vy, acx, acy = trajectory_profiles(xA, yA, xB, yB, T_leg, N)
    Fx, Fy = thrust_with_wind(vx, vy, acx, acy, m)
    power = np.abs(Fx * vx + Fy * vy)
    return np.trapz(power, t)


# =====================================================================
#  ROUND-TRIP ENERGY (motion + hover)
# =====================================================================

def roundtrip_energy(xA, yA, xB, yB, T_total, m, E_H, wind=False):
    T_leg = T_total / 2
    func = leg_energy_wind if wind else leg_energy_no_wind
    E_out = func(xA, yA, xB, yB, T_leg, m)
    E_ret = func(xB, yB, xA, yA, T_leg, m)
    return E_out + E_ret + E_H * T_total


# =====================================================================
#  MAIN
# =====================================================================

if __name__ == "__main__":
    # --- Inputs ---
    xA, yA = 0.0, 0.0
    xB, yB = 1.0, 2.0
    T_total = 1.0
    m = 1.0
    E_H = 1.0               # hovering power (energy per unit time)
    T_leg = T_total / 2

    D = np.sqrt((xB - xA)**2 + (yB - yA)**2)

    # =================================================================
    #  1. FIXED-TIME ANALYSIS  (T_total = 1.0 s)
    # =================================================================
    E_nw_out = leg_energy_no_wind(xA, yA, xB, yB, T_leg, m)
    E_nw_ret = leg_energy_no_wind(xB, yB, xA, yA, T_leg, m)
    E_nw_motion = E_nw_out + E_nw_ret
    E_nw_hover = E_H * T_total
    E_nw_total = E_nw_motion + E_nw_hover

    E_w_out = leg_energy_wind(xA, yA, xB, yB, T_leg, m)
    E_w_ret = leg_energy_wind(xB, yB, xA, yA, T_leg, m)
    E_w_motion = E_w_out + E_w_ret
    E_w_hover = E_H * T_total
    E_w_total = E_w_motion + E_w_hover

    E_diff = E_w_total - E_nw_total
    pct_energy = E_diff / E_nw_total * 100

    # =================================================================
    #  2. OPTIMAL TRAVEL-TIME SEARCH
    # =================================================================
    opt_nw = minimize_scalar(
        lambda T: roundtrip_energy(xA, yA, xB, yB, T, m, E_H, wind=False),
        bounds=(0.2, 20.0), method='bounded')
    opt_w = minimize_scalar(
        lambda T: roundtrip_energy(xA, yA, xB, yB, T, m, E_H, wind=True),
        bounds=(0.2, 20.0), method='bounded')

    T_opt_nw, E_opt_nw = opt_nw.x, opt_nw.fun
    T_opt_w, E_opt_w = opt_w.x, opt_w.fun
    T_shift = T_opt_w - T_opt_nw

    # energy at each other's optimal time (cross-comparison)
    E_nw_at_wopt = roundtrip_energy(xA, yA, xB, yB, T_opt_w, m, E_H, wind=False)
    E_w_at_nwopt = roundtrip_energy(xA, yA, xB, yB, T_opt_nw, m, E_H, wind=True)

    # =================================================================
    #  3. TIME SWEEP
    # =================================================================
    T_range = np.linspace(0.3, 6.0, 300)
    E_nw_sweep = np.array([roundtrip_energy(xA, yA, xB, yB, T, m, E_H, False)
                           for T in T_range])
    E_w_sweep = np.array([roundtrip_energy(xA, yA, xB, yB, T, m, E_H, True)
                          for T in T_range])

    # =================================================================
    #  FIGURE 1 — Thrust & Power Profiles (fixed T)
    # =================================================================
    t_prof, vx_o, vy_o, acx_o, acy_o = trajectory_profiles(
        xA, yA, xB, yB, T_leg)
    _, vx_r, vy_r, acx_r, acy_r = trajectory_profiles(
        xB, yB, xA, yA, T_leg)

    Fx_nw_o, Fy_nw_o = thrust_no_wind(acx_o, acy_o, m)
    Fx_w_o, Fy_w_o = thrust_with_wind(vx_o, vy_o, acx_o, acy_o, m)
    Fx_nw_r, Fy_nw_r = thrust_no_wind(acx_r, acy_r, m)
    Fx_w_r, Fy_w_r = thrust_with_wind(vx_r, vy_r, acx_r, acy_r, m)

    F_nw_o = np.sqrt(Fx_nw_o**2 + Fy_nw_o**2)
    F_w_o  = np.sqrt(Fx_w_o**2 + Fy_w_o**2)
    F_nw_r = np.sqrt(Fx_nw_r**2 + Fy_nw_r**2)
    F_w_r  = np.sqrt(Fx_w_r**2 + Fy_w_r**2)

    P_nw_o = np.abs(Fx_nw_o * vx_o + Fy_nw_o * vy_o)
    P_w_o  = np.abs(Fx_w_o  * vx_o + Fy_w_o  * vy_o)
    P_nw_r = np.abs(Fx_nw_r * vx_r + Fy_nw_r * vy_r)
    P_w_r  = np.abs(Fx_w_r  * vx_r + Fy_w_r  * vy_r)

    fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
    (a1, a2), (a3, a4) = axes

    a1.plot(t_prof, F_nw_o, 'b-', label='No Wind', lw=2)
    a1.plot(t_prof, F_w_o, 'r-', label='With Wind', lw=2)
    a1.set_xlabel('Time (s)'), a1.set_ylabel('|F_thrust| (N)')
    a1.set_title('Outbound (A→B): Thrust Magnitude')
    a1.legend(), a1.grid(True)

    a2.plot(t_prof, F_nw_r, 'b-', label='No Wind', lw=2)
    a2.plot(t_prof, F_w_r, 'r-', label='With Wind', lw=2)
    a2.set_xlabel('Time (s)'), a2.set_ylabel('|F_thrust| (N)')
    a2.set_title('Return (B→A): Thrust Magnitude')
    a2.legend(), a2.grid(True)

    a3.plot(t_prof, P_nw_o, 'b-', label='No Wind', lw=2)
    a3.plot(t_prof, P_w_o, 'r-', label='With Wind', lw=2)
    a3.fill_between(t_prof, P_nw_o, P_w_o, where=P_w_o > P_nw_o,
                    alpha=0.2, color='red', label='Extra cost')
    a3.fill_between(t_prof, P_nw_o, P_w_o, where=P_w_o < P_nw_o,
                    alpha=0.2, color='green', label='Wind assist')
    a3.set_xlabel('Time (s)'), a3.set_ylabel('|Power| (W)')
    a3.set_title('Outbound Power (red = extra, green = wind assist)')
    a3.legend(), a3.grid(True)

    a4.plot(t_prof, P_nw_r, 'b-', label='No Wind', lw=2)
    a4.plot(t_prof, P_w_r, 'r-', label='With Wind', lw=2)
    a4.fill_between(t_prof, P_nw_r, P_w_r, where=P_w_r > P_nw_r,
                    alpha=0.2, color='red', label='Extra cost')
    a4.fill_between(t_prof, P_nw_r, P_w_r, where=P_w_r < P_nw_r,
                    alpha=0.2, color='green', label='Wind assist')
    a4.set_xlabel('Time (s)'), a4.set_ylabel('|Power| (W)')
    a4.set_title('Return Power (red = extra, green = wind assist)')
    a4.legend(), a4.grid(True)

    plt.suptitle(f'Thrust & Power Profiles — Compensated Flight (T={T_total}s)',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig('thrust_power_profiles.png', dpi=150, bbox_inches='tight')

    # =================================================================
    #  FIGURE 2 — Energy vs Travel Time
    # =================================================================
    fig2, ax5 = plt.subplots(figsize=(11, 7))
    ax5.plot(T_range, E_nw_sweep, 'b-', label='No Wind', lw=2)
    ax5.plot(T_range, E_w_sweep, 'r-', label=f'Wind ({WIND_SPEED_MPH} mph W)', lw=2)

    ax5.scatter([T_opt_nw], [E_opt_nw], color='blue', s=120, zorder=5,
                edgecolors='black')
    ax5.scatter([T_opt_w], [E_opt_w], color='red', s=120, zorder=5,
                edgecolors='black')

    ax5.annotate(f'No-Wind Optimum\nT = {T_opt_nw:.3f} s\nE = {E_opt_nw:.2f} J',
                 (T_opt_nw, E_opt_nw), xytext=(40, 30),
                 textcoords='offset points', fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='blue'))
    ax5.annotate(f'Wind Optimum\nT = {T_opt_w:.3f} s\nE = {E_opt_w:.2f} J',
                 (T_opt_w, E_opt_w), xytext=(40, -40),
                 textcoords='offset points', fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='red'))

    # shade the time shift
    ax5.axvline(T_opt_nw, color='blue', ls='--', alpha=0.4)
    ax5.axvline(T_opt_w, color='red', ls='--', alpha=0.4)
    ymin_shade = min(E_opt_nw, E_opt_w) * 0.95
    ax5.fill_betweenx([ymin_shade, ymin_shade + 1],
                      T_opt_nw, T_opt_w, alpha=0.15, color='purple')
    ax5.text((T_opt_nw + T_opt_w) / 2, ymin_shade + 0.5,
             f'ΔT = {T_shift:+.3f} s', ha='center', fontsize=10,
             color='purple', fontweight='bold')

    ax5.set_xlabel('Total Round-Trip Time T (s)', fontsize=12)
    ax5.set_ylabel('Total Energy (J)', fontsize=12)
    ax5.set_title('Energy vs Travel Time — Maintaining Same Flight Path',
                  fontsize=13)
    ax5.legend(fontsize=11), ax5.grid(True)
    ax5.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig('energy_vs_time.png', dpi=150)

    # =================================================================
    #  FIGURE 3 — Energy Bar Chart (fixed T)
    # =================================================================
    fig3, ax6 = plt.subplots(figsize=(10, 6))
    labels = ['Motion\n(Out)', 'Motion\n(Return)', 'Total\nMotion',
              'Hover', 'Grand\nTotal']
    nw_vals = [E_nw_out, E_nw_ret, E_nw_motion, E_nw_hover, E_nw_total]
    w_vals  = [E_w_out,  E_w_ret,  E_w_motion,  E_w_hover,  E_w_total]

    x_pos = np.arange(len(labels))
    width = 0.35
    bars1 = ax6.bar(x_pos - width/2, nw_vals, width,
                    label='No Wind', color='steelblue')
    bars2 = ax6.bar(x_pos + width/2, w_vals, width,
                    label=f'Wind Compensated', color='indianred')
    ax6.set_ylabel('Energy (J)')
    ax6.set_title(f'Energy at T = {T_total} s (drone maintains ideal path)')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(labels)
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)
    for bar in bars1:
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig('energy_comparison.png', dpi=150)

    # =================================================================
    #  CONSOLE OUTPUT
    # =================================================================
    print("=" * 66)
    print("  WIND-COMPENSATED FLIGHT ANALYSIS")
    print("  Drone adapts thrust to maintain ideal trajectory")
    print("  Infinite energy supply assumed")
    print("=" * 66)
    print(f"\nDrag : F = -½ρCdA|v_rel|v_rel")
    print(f"       ρ = {RHO} kg/m³,  Cd = {CD},  A = {A_CROSS} m²")
    print(f"Wind : {WIND_SPEED_MPH} mph West = {WIND_SPEED_MS:.4f} m/s in -x")
    print(f"Drone: m = {m} kg,  D = {D:.4f} m")
    print(f"Path : A=({xA},{yA}) → B=({xB},{yB}) → A")

    print(f"\n{'─'*66}")
    print(f"  ENERGY AT FIXED T = {T_total} s  (same path, same time)")
    print(f"{'─'*66}")
    print(f"  {'Metric':<40} {'No Wind':>10} {'With Wind':>10}")
    print(f"  {'─'*60}")
    print(f"  {'Outbound motion energy (J)':<40} {E_nw_out:>10.4f} {E_w_out:>10.4f}")
    print(f"  {'Return motion energy (J)':<40} {E_nw_ret:>10.4f} {E_w_ret:>10.4f}")
    print(f"  {'Total motion energy (J)':<40} {E_nw_motion:>10.4f} {E_w_motion:>10.4f}")
    print(f"  {'Hover energy (J)':<40} {E_nw_hover:>10.4f} {E_w_hover:>10.4f}")
    print(f"  {'TOTAL energy (J)':<40} {E_nw_total:>10.4f} {E_w_total:>10.4f}")
    print(f"  {'Energy difference (J)':<40} {'':>10} {E_diff:>+10.4f}")
    print(f"  {'Energy change (%)':<40} {'':>10} {pct_energy:>+10.2f}%")

    print(f"\n{'─'*66}")
    print(f"  OPTIMAL TRAVEL TIME  (minimise total energy)")
    print(f"{'─'*66}")
    print(f"  {'Metric':<40} {'No Wind':>10} {'With Wind':>10}")
    print(f"  {'─'*60}")
    print(f"  {'Optimal T_total (s)':<40} {T_opt_nw:>10.4f} {T_opt_w:>10.4f}")
    print(f"  {'Minimum total energy (J)':<40} {E_opt_nw:>10.4f} {E_opt_w:>10.4f}")
    print(f"  {'Time shift ΔT (s)':<40} {'':>10} {T_shift:>+10.4f}")
    pct_time = T_shift / T_opt_nw * 100
    print(f"  {'Time change (%)':<40} {'':>10} {pct_time:>+10.2f}%")
    E_opt_diff = E_opt_w - E_opt_nw
    pct_opt_e = E_opt_diff / E_opt_nw * 100
    print(f"  {'Energy increase at optimum (J)':<40} {'':>10} {E_opt_diff:>+10.4f}")
    print(f"  {'Energy increase at optimum (%)':<40} {'':>10} {pct_opt_e:>+10.2f}%")

    print(f"\n  Cross-comparison:")
    print(f"    No-wind energy at wind-optimal T:  {E_nw_at_wopt:.4f} J")
    print(f"    Wind energy at no-wind-optimal T:  {E_w_at_nwopt:.4f} J")

    print(f"\n{'='*66}")
    print(f"  SUMMARY")
    print(f"{'='*66}")
    print(f"  At fixed T = {T_total}s:")
    if E_diff > 0:
        print(f"    Wind INCREASES energy by {E_diff:.4f} J ({pct_energy:+.2f}%)")
    else:
        print(f"    Wind DECREASES energy by {abs(E_diff):.4f} J ({pct_energy:+.2f}%)")

    print(f"\n  Outbound A→B (going east, against wind component):")
    out_diff = E_w_out - E_nw_out
    print(f"    Energy change: {out_diff:+.4f} J ({out_diff/E_nw_out*100:+.2f}%)")
    print(f"\n  Return B→A (going west, with wind component):")
    ret_diff = E_w_ret - E_nw_ret
    print(f"    Energy change: {ret_diff:+.4f} J ({ret_diff/E_nw_ret*100:+.2f}%)")

    print(f"\n  Energy-optimal travel time:")
    print(f"    No wind: T* = {T_opt_nw:.4f} s  →  E* = {E_opt_nw:.4f} J")
    print(f"    Wind:    T* = {T_opt_w:.4f} s  →  E* = {E_opt_w:.4f} J")
    if T_shift > 0:
        print(f"    Wind makes optimal trip {T_shift:.4f} s LONGER ({pct_time:+.2f}%)")
    else:
        print(f"    Wind makes optimal trip {abs(T_shift):.4f} s SHORTER ({pct_time:+.2f}%)")
    print(f"    Minimum energy cost of wind: +{E_opt_diff:.4f} J ({pct_opt_e:+.2f}%)")
    print(f"{'='*66}")
