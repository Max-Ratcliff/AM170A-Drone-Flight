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
        self, physics_model: DronePhysics, segment_vector: np.ndarray
    ) -> None:
        """
        Plot U-shaped Energy vs Velocity curve and mark the global minimum.

        Args:
            physics_model: DronePhysics instance for energy calculations.
            segment_vector: 2D segment vector in meters.
        """
        distance = float(np.linalg.norm(segment_vector))
        velocities = np.linspace(1.0, 30.0, 200)
        energies = [physics_model.calculate_energy(segment_vector, v) for v in velocities]

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

        ax.set_xlabel("Ground Velocity [m/s]", fontsize=14)
        ax.set_ylabel("Energy [Joules]", fontsize=14)
        wind_str = f"[{physics_model.wind[0]:.1f}, {physics_model.wind[1]:.1f}] m/s"
        ax.set_title(
            f"Energy vs Velocity (d = {distance:.0f} m, wind = {wind_str})", fontsize=16
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
        wind_vector: np.ndarray = np.array([0.0, 0.0]),
    ) -> None:
        """
        Plot 2D waypoints with naive vs optimized route comparison (1x2 subplot).

        Args:
            waypoints: N x 2 array of (x, y) coordinates.
            optimal_order: Ordered list of waypoint indices (optimized route).
            optimal_energy: Total energy cost of optimized route (J).
            naive_order: Ordered list of waypoint indices (naive 0..N-1 route).
            naive_energy: Total energy cost of naive route (J).
            wind_vector: 2D wind vector (w_x, w_y) in m/s.
        """
        fig, (ax_route, ax_bars) = plt.subplots(1, 2, figsize=(14, 6))

        # Add background wind field if wind is non-zero
        wind_speed = float(np.linalg.norm(wind_vector))
        if wind_speed > 0.1:
            x_min, x_max = waypoints[:, 0].min(), waypoints[:, 0].max()
            y_min, y_max = waypoints[:, 1].min(), waypoints[:, 1].max()
            # Pad the range for arrows
            pad_x = (x_max - x_min) * 0.1 + 50
            pad_y = (y_max - y_min) * 0.1 + 50
            
            x_grid = np.linspace(x_min - pad_x, x_max + pad_x, 10)
            y_grid = np.linspace(y_min - pad_y, y_max + pad_y, 10)
            X, Y = np.meshgrid(x_grid, y_grid)
            U = np.full(X.shape, wind_vector[0])
            V = np.full(X.shape, wind_vector[1])
            
            ax_route.quiver(
                X, Y, U, V, 
                color="gray", 
                alpha=0.2, 
                pivot="middle",
                label=f"Wind ({wind_speed:.1f} m/s)"
            )

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

        # Optimized route (solid, blue)
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

        # Label each point with its naive order index (0, 1, 2, ...)
        for i in range(len(waypoints)):
            ax_route.annotate(
                str(i),
                (waypoints[i, 0], waypoints[i, 1]),
                xytext=(0, 14),
                textcoords="offset points",
                ha="center",
                fontsize=11,
                zorder=12,
            )

        # Start/End marker (closed loop: route returns to start)
        start_idx = optimal_order[0]
        ax_route.scatter(
            waypoints[start_idx, 0],
            waypoints[start_idx, 1],
            marker="o",
            s=200,
            facecolors="none",
            edgecolors="green",
            linewidths=3,
            label="Start/End",
            zorder=11,
        )

        legend_elements = [
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
                label="Start/End",
            ),
        ]
        ax_route.legend(handles=legend_elements)
        ax_route.set_xlabel("x [m]", fontsize=14)
        ax_route.set_ylabel("y [m]", fontsize=14)
        ax_route.set_title("Drone Route Comparison", fontsize=14)
        ax_route.set_aspect("equal")
        ax_route.margins(0.12)
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
