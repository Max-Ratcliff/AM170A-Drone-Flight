"""Visualization of energy curves and optimal routes."""

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from physics import DronePhysics

if TYPE_CHECKING:
    pass

PLOTS_DIR = Path(__file__).resolve().parent / "plots"


class Visualizer:
    """Plots energy vs velocity curves and 2D route maps."""

    def plot_energy_curve(
        self, physics_model: DronePhysics, distance: float
    ) -> None:
        """
        Plot U-shaped Energy vs Velocity curve and mark the global minimum.

        Args:
            physics_model: DronePhysics instance for energy calculations.
            distance: Segment length in meters (e.g., 1000).
        """
        velocities = np.linspace(1.0, 30.0, 200)
        energies = [physics_model.calculate_energy(distance, v) for v in velocities]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(velocities, energies, "b-", linewidth=2, label="Energy vs Velocity")

        # Find and mark minimum
        idx_min = np.argmin(energies)
        v_opt = velocities[idx_min]
        e_min = energies[idx_min]
        ax.scatter(
            [v_opt],
            [e_min],
            color="red",
            s=100,
            zorder=5,
            label=f"v_opt = {v_opt:.2f} m/s",
        )
        ax.axvline(v_opt, color="red", linestyle="--", alpha=0.5)

        ax.set_xlabel("Velocity [m/s]", fontsize=14)
        ax.set_ylabel("Energy [Joules]", fontsize=14)
        ax.set_title(
            f"Segment Energy vs Velocity (d = {distance} m)", fontsize=16
        )
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.6)
        fig.tight_layout()
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / "energy_curve.png", dpi=150)
        plt.close(fig)

    def plot_route(
        self,
        waypoints: np.ndarray,
        optimal_order: list[int],
        optimal_energy: float,
        naive_order: list[int],
        naive_energy: float,
    ) -> None:
        """
        Plot 2D waypoints with naive vs optimized route comparison (1x2 subplot).

        Args:
            waypoints: N x 2 array of (x, y) coordinates.
            optimal_order: Ordered list of waypoint indices (optimized route).
            optimal_energy: Total energy cost of optimized route (J).
            naive_order: Ordered list of waypoint indices (naive 0..N-1 route).
            naive_energy: Total energy cost of naive route (J).
        """
        fig, (ax_route, ax_bars) = plt.subplots(1, 2, figsize=(14, 6))

        num_arrows_per_segment = 4
        arrow_style = "->,head_width=0.15,head_length=0.1"

        def draw_route(
            ax: Axes,
            order: list[int],
            color: str,
            linestyle: str,
            lw: float,
        ) -> None:
            for k in range(len(order)):
                i = order[k]
                j = order[(k + 1) % len(order)]
                xi, yi = waypoints[i, 0], waypoints[i, 1]
                xj, yj = waypoints[j, 0], waypoints[j, 1]
                for t in np.linspace(0, 1, num_arrows_per_segment + 1)[
                    :-1
                ]:
                    t_next = t + 1.0 / (num_arrows_per_segment + 1)
                    x0 = xi + t * (xj - xi)
                    y0 = yi + t * (yj - yi)
                    x1 = xi + t_next * (xj - xi)
                    y1 = yi + t_next * (yj - yi)
                    ax.annotate(
                        "",
                        xy=(x1, y1),
                        xytext=(x0, y0),
                        arrowprops=dict(
                            arrowstyle=arrow_style,
                            color=color,
                            linestyle=linestyle,
                            lw=lw,
                            alpha=0.7 if linestyle == "--" else 1.0,
                        ),
                    )

        # Naive route first (dashed, red)
        draw_route(ax_route, naive_order, "red", "--", 1.2)
        # Optimized route on top (solid, blue)
        draw_route(ax_route, optimal_order, "blue", "-", 1.5)

        # Scatter waypoints on top of lines (higher zorder)
        ax_route.scatter(
            waypoints[:, 0],
            waypoints[:, 1],
            c="steelblue",
            s=120,
            zorder=10,
            edgecolors="black",
            linewidths=1.5,
        )

        # Start/End markers
        start_idx = optimal_order[0]
        end_idx = optimal_order[-1]
        ax_route.scatter(
            waypoints[start_idx, 0],
            waypoints[start_idx, 1],
            marker="o",
            s=200,
            facecolors="none",
            edgecolors="green",
            linewidths=3,
            label="Start",
            zorder=11,
        )
        ax_route.scatter(
            waypoints[end_idx, 0],
            waypoints[end_idx, 1],
            marker="s",
            s=200,
            facecolors="none",
            edgecolors="darkorange",
            linewidths=3,
            label="End",
            zorder=11,
        )

        legend_elements = [
            Line2D(
                [0],
                [0],
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label="Naive Route",
            ),
            Line2D(
                [0],
                [0],
                color="blue",
                linestyle="-",
                linewidth=2,
                label="Optimized Route",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="w",
                markeredgecolor="green",
                markeredgewidth=2,
                markersize=10,
                linestyle="None",
                label="Start",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="w",
                markeredgecolor="darkorange",
                markeredgewidth=2,
                markersize=10,
                linestyle="None",
                label="End",
            ),
        ]
        ax_route.legend(handles=legend_elements)
        ax_route.set_xlabel("x [m]", fontsize=14)
        ax_route.set_ylabel("y [m]", fontsize=14)
        ax_route.set_title("Drone Route Comparison", fontsize=14)
        ax_route.set_aspect("equal")
        ax_route.grid(True, alpha=0.6)

        # Bar chart of energy comparison
        bars = ax_bars.bar(
            ["Naive", "Optimized"],
            [naive_energy, optimal_energy],
            color=["red", "blue"],
            edgecolor="black",
            linewidth=1.2,
        )
        bars[0].set_alpha(0.7)
        ax_bars.set_ylabel("Energy [J]", fontsize=14)
        ax_bars.set_title("Route Energy Comparison", fontsize=14)
        ax_bars.tick_params(axis="both", labelsize=12)
        for bar in bars:
            ax_bars.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.0f}",
                ha="center",
                va="bottom",
                fontsize=11,
            )
        ax_bars.grid(True, alpha=0.6, axis="y")

        fig.tight_layout()
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / "route_map.png", dpi=150)
        plt.close(fig)
