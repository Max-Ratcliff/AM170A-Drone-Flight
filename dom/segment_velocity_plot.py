"""
Optimal Velocity per Segment Visualization

Generates a color-coded route map where each leg of the trip is colored
by its optimal ground velocity, plus a bar chart showing the optimal
velocity for every segment with the wind-relative heading angle.

Usage:
    python dom/segment_velocity_plot.py                  # 5 random waypoints, no wind
    python dom/segment_velocity_plot.py --test           # fixed test set with wind
    python dom/segment_velocity_plot.py -n 6 -s 42 -w 5.0 -2.0
    python dom/segment_velocity_plot.py -m nn            # nearest-neighbor TSP
"""

import sys
from pathlib import Path

# allow imports from final_project/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "final_project"))

import argparse

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.lines import Line2D

from optimizer import RoutingOptimizer
from params import (SimulationConfig, get_default_params,
                    get_default_sim_config, get_test_sim_config)
from physics import DronePhysics
from targets import Targets

PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def compute_route_data(waypoints, order, optimal_velocities, energy_matrix, wind):
    """Compute per-segment velocity, energy, heading, and wind-relative angle."""
    segments = []
    for k in range(len(order)):
        i = order[k]
        j = order[(k + 1) % len(order)]
        seg_vec = waypoints[j] - waypoints[i]
        distance = float(np.linalg.norm(seg_vec))
        v_opt = optimal_velocities.get((i, j), 0.0)
        energy = energy_matrix[i, j]

        # heading angle of segment (degrees from +x axis)
        heading = np.degrees(np.arctan2(seg_vec[1], seg_vec[0]))

        # angle between travel direction and wind
        if distance > 0 and np.linalg.norm(wind) > 0.1:
            seg_unit = seg_vec / distance
            wind_unit = wind / np.linalg.norm(wind)
            cos_angle = np.clip(np.dot(seg_unit, wind_unit), -1, 1)
            wind_angle = np.degrees(np.arccos(cos_angle))
        else:
            wind_angle = None

        segments.append({
            "from": i, "to": j,
            "distance": distance,
            "v_opt": v_opt,
            "energy": energy,
            "heading": heading,
            "wind_angle": wind_angle,
        })
    return segments


def plot_velocity_route(waypoints, order, segments, wind, save_path):
    """Color-coded route map where each leg is colored by optimal velocity."""
    velocities = [s["v_opt"] for s in segments]
    v_min, v_max = min(velocities), max(velocities)

    # use a diverging colormap so slow=blue, fast=red
    if v_max - v_min < 0.01:
        norm = mcolors.Normalize(vmin=v_min - 1, vmax=v_max + 1)
    else:
        norm = mcolors.Normalize(vmin=v_min, vmax=v_max)
    cmap = cm.coolwarm

    fig, ax = plt.subplots(figsize=(10, 8))

    # wind field background
    wind_speed = float(np.linalg.norm(wind))
    if wind_speed > 0.1:
        x_min, x_max = waypoints[:, 0].min(), waypoints[:, 0].max()
        y_min, y_max = waypoints[:, 1].min(), waypoints[:, 1].max()
        pad_x = (x_max - x_min) * 0.1 + 50
        pad_y = (y_max - y_min) * 0.1 + 50
        xg = np.linspace(x_min - pad_x, x_max + pad_x, 8)
        yg = np.linspace(y_min - pad_y, y_max + pad_y, 8)
        X, Y = np.meshgrid(xg, yg)
        U = np.full(X.shape, wind[0])
        V = np.full(X.shape, wind[1])
        ax.quiver(X, Y, U, V, color="gray", alpha=0.15, pivot="middle")

    # draw each segment colored by velocity
    for seg in segments:
        i, j = seg["from"], seg["to"]
        color = cmap(norm(seg["v_opt"]))
        xi, yi = waypoints[i]
        xj, yj = waypoints[j]
        dx, dy = xj - xi, yj - yi

        ax.annotate(
            "", xy=(xj, yj), xytext=(xi, yi),
            arrowprops=dict(
                arrowstyle="->,head_width=0.25,head_length=0.15",
                color=color, lw=2.5,
            ),
        )

        # velocity label at midpoint
        mx, my = (xi + xj) / 2, (yi + yj) / 2
        ax.text(mx, my, f"{seg['v_opt']:.1f}",
                fontsize=8, ha="center", va="bottom",
                color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, ec="none"))

    # waypoints
    ax.scatter(waypoints[:, 0], waypoints[:, 1],
               c="steelblue", s=140, zorder=10, edgecolors="black", linewidths=1.5)
    for idx in range(len(waypoints)):
        ax.annotate(str(idx), (waypoints[idx, 0], waypoints[idx, 1]),
                     xytext=(0, 14), textcoords="offset points",
                     ha="center", fontsize=11, zorder=12)

    # start marker
    si = order[0]
    ax.scatter(waypoints[si, 0], waypoints[si, 1],
               marker="o", s=220, facecolors="none",
               edgecolors="green", linewidths=3, zorder=11)

    # colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Optimal Ground Velocity [m/s]", fontsize=12)

    wind_str = f"wind = [{wind[0]:.1f}, {wind[1]:.1f}] m/s" if wind_speed > 0.1 else "no wind"
    ax.set_title(f"Optimal Velocity per Segment ({wind_str})", fontsize=14)
    ax.set_xlabel("x [m]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)
    ax.set_aspect("equal")
    ax.margins(0.12)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_velocity_bars(segments, wind, save_path):
    """Bar chart of optimal velocity per segment, colored by wind-relative angle."""
    n = len(segments)
    labels = [f"{s['from']}\u2192{s['to']}" for s in segments]
    velocities = [s["v_opt"] for s in segments]
    distances = [s["distance"] for s in segments]
    wind_angles = [s["wind_angle"] for s in segments]
    wind_speed = float(np.linalg.norm(wind))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, n * 0.9), 9),
                                    gridspec_kw={"height_ratios": [3, 2]})

    # top: velocity bars colored by wind angle
    if wind_speed > 0.1 and all(a is not None for a in wind_angles):
        norm = mcolors.Normalize(vmin=0, vmax=180)
        cmap = cm.RdYlGn  # green=tailwind(0°), red=headwind(180°)
        colors = [cmap(norm(a)) for a in wind_angles]
    else:
        colors = ["steelblue"] * n

    bars = ax1.bar(range(n), velocities, color=colors, edgecolor="black", linewidth=0.8)
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(labels, fontsize=9, rotation=45 if n > 8 else 0)
    ax1.set_ylabel("Optimal Velocity [m/s]", fontsize=12)
    ax1.set_title("Optimal Ground Velocity per Route Segment", fontsize=14)
    ax1.grid(True, alpha=0.4, axis="y")

    # value labels on bars
    for bar, v in zip(bars, velocities):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    # colorbar for wind angle
    if wind_speed > 0.1 and all(a is not None for a in wind_angles):
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax1, pad=0.02)
        cbar.set_label("Angle to Wind [deg]\n(0°=tailwind, 180°=headwind)", fontsize=10)

    # bottom: segment distance bars
    ax2.bar(range(n), distances, color="lightcoral", edgecolor="black", linewidth=0.8)
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(labels, fontsize=9, rotation=45 if n > 8 else 0)
    ax2.set_ylabel("Distance [m]", fontsize=12)
    ax2.set_title("Segment Distances", fontsize=14)
    ax2.grid(True, alpha=0.4, axis="y")
    for i, d in enumerate(distances):
        ax2.text(i, d, f"{d:.0f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Optimal velocity per segment visualization")
    parser.add_argument("-n", "--num-targets", type=int, default=5)
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument("--test", action="store_true",
                        help="Use fixed test waypoint set (includes wind)")
    parser.add_argument("-w", "--wind", type=float, nargs=2,
                        default=[0.0, 0.0], metavar=("WX", "WY"))
    parser.add_argument("-d", "--distribution",
                        choices=["uniform", "clustered", "grid"], default="uniform")
    parser.add_argument("-m", "--method", choices=["brute", "nn", "held_karp"],
                        default="brute")
    args = parser.parse_args()

    if args.test:
        config = get_test_sim_config()
    else:
        config = SimulationConfig(
            num_targets=args.num_targets,
            seed=args.seed,
            wind_vector=tuple(args.wind),
            distribution=args.distribution,
        )

    params = get_default_params()
    targets = Targets(
        num_targets=config.num_targets,
        bounds=config.bounds,
        waypoint_set=config.waypoint_set,
        distribution=config.distribution,
        seed=config.seed,
    )
    waypoints = targets.generate_waypoints()

    physics = DronePhysics(params, wind_vector=config.wind_vector)
    optimizer = RoutingOptimizer(physics)
    energy_matrix, optimal_velocities = optimizer.build_energy_matrix(waypoints)

    method_map = {"nn": "nearest_neighbor", "held_karp": "held_karp", "brute": "brute"}
    optimal_order = optimizer.solve_tsp(energy_matrix, method=method_map[args.method])

    wind = np.array(config.wind_vector)
    segments = compute_route_data(waypoints, optimal_order, optimal_velocities,
                                  energy_matrix, wind)

    # print summary
    print("=== Optimal Velocity per Segment ===")
    print(f"Wind: [{wind[0]:.1f}, {wind[1]:.1f}] m/s")
    print(f"Route: {optimal_order}")
    print(f"{'Leg':<10} {'Dist (m)':<10} {'V_opt (m/s)':<12} {'Wind Angle':<12}")
    print("-" * 44)
    for s in segments:
        wa = f"{s['wind_angle']:.1f}°" if s['wind_angle'] is not None else "N/A"
        print(f"{s['from']}->{s['to']:<6} {s['distance']:<10.1f} {s['v_opt']:<12.2f} {wa:<12}")

    plot_velocity_route(waypoints, optimal_order, segments, wind,
                        PLOTS_DIR / "segment_velocity_route.png")
    plot_velocity_bars(segments, wind,
                       PLOTS_DIR / "segment_velocity_bars.png")


if __name__ == "__main__":
    main()
