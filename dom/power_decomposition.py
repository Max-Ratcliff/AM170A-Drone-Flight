"""
Power Component Decomposition Visualization

Plots the three terms of P(v) = c1 + c2*v³ + c3/v as separate curves,
showing where blade profile, parasitic drag, and induced power each dominate.
Marks the optimal velocity where total power is minimized.

Usage:
    python dom/power_decomposition.py                # default params
    python dom/power_decomposition.py --test         # test config label
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "final_project"))

import argparse

import matplotlib.pyplot as plt
import numpy as np

from params import get_default_params

PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def main():
    parser = argparse.ArgumentParser(
        description="Power component decomposition plot")
    parser.add_argument("--test", action="store_true",
                        help="Label plot as test configuration")
    parser.add_argument("--v-max", type=float, default=30.0,
                        help="Maximum velocity to plot (m/s)")
    args = parser.parse_args()

    params = get_default_params()
    c1, c2, c3 = params.c1, params.c2, params.c3

    v = np.linspace(0.5, args.v_max, 500)

    p_blade = np.full_like(v, c1)        # blade profile (constant)
    p_parasitic = c2 * v**3              # parasitic drag
    p_induced = c3 / v                   # induced power
    p_total = p_blade + p_parasitic + p_induced

    # find optimal velocity (minimum total power)
    i_opt = int(np.argmin(p_total))
    v_opt = v[i_opt]
    p_opt = p_total[i_opt]

    # --- stacked area + line plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9),
                                    gridspec_kw={"height_ratios": [3, 2]})

    # top: stacked area
    ax1.fill_between(v, 0, p_blade, alpha=0.3, color="steelblue", label=f"Blade profile: c₁ = {c1}")
    ax1.fill_between(v, p_blade, p_blade + p_induced, alpha=0.3, color="orange",
                     label=f"Induced: c₃/v (c₃ = {c3})")
    ax1.fill_between(v, p_blade + p_induced, p_blade + p_induced + p_parasitic,
                     alpha=0.3, color="crimson", label=f"Parasitic drag: c₂v³ (c₂ = {c2})")
    ax1.plot(v, p_total, "k-", linewidth=2.5, label="Total P(v)")
    ax1.axvline(v_opt, color="green", linestyle="--", linewidth=1.5, alpha=0.8)
    ax1.plot(v_opt, p_opt, "o", color="green", markersize=10, zorder=10,
             label=f"P_min = {p_opt:.0f} W at v = {v_opt:.1f} m/s")

    ax1.set_xlabel("Airspeed v [m/s]", fontsize=12)
    ax1.set_ylabel("Power [W]", fontsize=12)
    ax1.set_title("Power Decomposition: P(v) = c₁ + c₂v³ + c₃/v", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.set_xlim(v[0], v[-1])
    ax1.set_ylim(0, min(p_total.max() * 1.3, p_total[i_opt] * 5))
    ax1.grid(True, alpha=0.4)

    # bottom: fraction of total power from each component
    frac_blade = p_blade / p_total * 100
    frac_parasitic = p_parasitic / p_total * 100
    frac_induced = p_induced / p_total * 100

    ax2.stackplot(v, frac_blade, frac_induced, frac_parasitic,
                  colors=["steelblue", "orange", "crimson"], alpha=0.5,
                  labels=["Blade profile", "Induced", "Parasitic drag"])
    ax2.axvline(v_opt, color="green", linestyle="--", linewidth=1.5, alpha=0.8)
    ax2.set_xlabel("Airspeed v [m/s]", fontsize=12)
    ax2.set_ylabel("Fraction of Total Power [%]", fontsize=12)
    ax2.set_title("Power Component Fractions", fontsize=14)
    ax2.set_xlim(v[0], v[-1])
    ax2.set_ylim(0, 100)
    ax2.legend(loc="center right", fontsize=10)
    ax2.grid(True, alpha=0.4)

    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PLOTS_DIR / "power_decomposition.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Drone params: c1={c1}, c2={c2}, c3={c3}")
    print(f"Optimal airspeed: {v_opt:.2f} m/s  (P_min = {p_opt:.1f} W)")
    print(f"At v_opt: blade={p_blade[i_opt]:.1f} W, "
          f"induced={p_induced[i_opt]:.1f} W, "
          f"parasitic={p_parasitic[i_opt]:.1f} W")
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
