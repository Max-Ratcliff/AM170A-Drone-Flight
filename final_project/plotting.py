"""Visualization of energy curves and optimal routes."""

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from params import get_default_params
from physics import DronePhysics

if TYPE_CHECKING:
    pass

PLOTS_DIR = Path(__file__).resolve().parent / "plots"


class Visualizer:
    """Plots energy vs time curves and 2D route maps."""

    def plot_energy_curve(
        self, base_physics: DronePhysics, segment_vector: np.ndarray
    ) -> None:
        """
        Plot a series of E(T) curves as shown in the project sketch:
        1. Hover energy baseline (P_hover * T)
        2. No drag case (analytical/ideal)
        3. Real physics with varying drag (lambda > 0)
        """
        d = float(np.linalg.norm(segment_vector))
        # Use a generous time range for the U-shape
        T_min_phys, T_max_phys = base_physics.feasible_time_bounds(d)
        Ts = np.linspace(T_min_phys * 0.8, T_max_phys * 1.5, 300)

        fig, ax = plt.subplots(figsize=(10, 6.5))

        # Hover Energy Baseline: E_H * T
        hover_energy = base_physics.p.hover_power * Ts

        # No Drag Case (lambda = 0): Analytical version
        # Energy = P_hover*T + Work(acceleration)
        # For a stop-to-stop parabolic profile, Work = m * (1.5 d / T)^2
        no_drag_energy = hover_energy + base_physics.p.mass * (1.5 * d / Ts) ** 2
        ax.plot(Ts, no_drag_energy, "g-", alpha=0.5, label="No Drag (Ideal)")

        # Series of Drag Coefficients (lambda > 0)
        # Original drag
        current_energy = [base_physics.segment_energy(segment_vector, T) for T in Ts]
        ax.plot(
            Ts,
            current_energy,
            "b-",
            linewidth=2,
            label=f"Current Drag ($C={base_physics.p.drag_coeff}$)",
        )

        # Mark the global minimum for current drag
        idx = np.argmin(current_energy)
        ax.scatter(
            [Ts[idx]],
            [current_energy[idx]],
            color="red",
            s=80,
            zorder=5,
            label=f"$T_{{opt}} = {Ts[idx]:.1f}s$",
        )

        # High drag scenarios (visualizing sensitivity)
        for multiplier, style in [(2.0, ":"), (4.0, "-.")]:
            params_high = get_default_params(
                mass=base_physics.p.mass,
                drag_coeff=base_physics.p.drag_coeff * multiplier,
                hover_power=base_physics.p.hover_power,
            )
            phys_high = DronePhysics(params_high, wind_vector=tuple(base_physics.wind))
            E_high = [phys_high.segment_energy(segment_vector, T) for T in Ts]
            ax.plot(
                Ts,
                E_high,
                color="blue",
                linestyle=style,
                alpha=0.4,
                label=f"Increased Drag ($\\lambda={multiplier}C$)",
            )

        ax.set_xlabel("Segment Time $T$ [s]", fontsize=18)
        ax.set_ylabel("Total Energy $E(T)$ [Joules]", fontsize=18)

        wind_str = f"[{base_physics.wind[0]:.1f}, {base_physics.wind[1]:.1f}] m/s"
        ax.set_title(
            f"Energy VS Drag Analysis (d = {d:.0f}m, wind = {wind_str})",
            fontsize=22,
        )

        ax.set_ylim(
            0, float(np.percentile(current_energy, 95) * 1.2)
        )  # Avoid zooming out too much due to small T
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=14)

        fig.tight_layout()
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / "energy_curve.png", dpi=200)
        plt.close(fig)

    # ---------------- Grid route comparison ----------------
    def plot_route_grid(
        self,
        waypoints: np.ndarray,
        orders: Dict[str, List[int]],
        energies: Dict[str, float],
        wind_vector: np.ndarray = np.array([0.0, 0.0]),
        filename: str = "route_comparison_grid.png",
    ) -> None:
        """
        Plots a grid of route comparisons (1x2, 1x3, or 2x2).
        """
        n_plots = len(orders)
        if n_plots <= 2:
            rows, cols = 1, 2
        elif n_plots == 3:
            rows, cols = 1, 3
        else:
            rows, cols = 2, 2

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))

        if n_plots == 1:
            axes_list = [axes]
        else:
            axes_list = axes.flatten()

        def draw_route(ax: Axes, order: List[int], title: str, energy: float) -> None:
            # Add background wind field
            wind_speed = float(np.linalg.norm(wind_vector))
            if wind_speed > 0.1:
                x_min, x_max = waypoints[:, 0].min(), waypoints[:, 0].max()
                y_min, y_max = waypoints[:, 1].min(), waypoints[:, 1].max()
                pad_x = (x_max - x_min) * 0.1 + 50
                pad_y = (y_max - y_min) * 0.1 + 50

                x_grid = np.linspace(x_min - pad_x, x_max + pad_x, 8)
                y_grid = np.linspace(y_min - pad_y, y_max + pad_y, 8)
                X, Y = np.meshgrid(x_grid, y_grid)
                U = np.full(X.shape, wind_vector[0])
                V = np.full(X.shape, wind_vector[1])

                ax.quiver(
                    X, Y, U, V, color="gray", alpha=0.15, pivot="middle", zorder=1
                )

            ax.scatter(
                waypoints[:, 0],
                waypoints[:, 1],
                s=80,
                color="#4C72B0",
                edgecolor="black",
                linewidth=1,
                zorder=3,
            )

            ax.scatter(
                waypoints[order[0], 0],
                waypoints[order[0], 1],
                s=180,
                facecolor="none",
                edgecolor="green",
                linewidth=2.5,
                zorder=4,
            )

            for idx, (x, y) in enumerate(waypoints):
                ax.text(x + 10, y + 10, str(idx), fontsize=10, zorder=5)

            for k in range(len(order)):
                i, j = order[k], order[(k + 1) % len(order)]
                xi, yi = waypoints[i, 0], waypoints[i, 1]
                xj, yj = waypoints[j, 0], waypoints[j, 1]

                ax.plot(
                    [xi, xj],
                    [yi, yj],
                    linestyle="--",
                    linewidth=1.5,
                    color="blue",
                    alpha=0.8,
                    zorder=2,
                )
                ax.annotate(
                    "",
                    xy=(xj, yj),
                    xytext=(xi, yi),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="blue", alpha=0.7),
                    zorder=2,
                )

            ax.set_title(f"{title}\nEnergy: {energy:.0f} J", fontsize=18)
            ax.set_xlabel("x [m]", fontsize=14)
            ax.set_ylabel("y [m]", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal", adjustable="box")

        display_names = {
            "Naive": "Naive (Sequential)",
            "nearest_neighbor": "Nearest Neighbor",
            "nn_2opt": "NN + 2-opt Improvement",
            "held_karp": "Held-Karp (Exact)",
            "brute": "Brute Force (Exact)",
        }

        for idx, (method, order) in enumerate(orders.items()):
            title = display_names.get(method, method)
            draw_route(axes_list[idx], order, title, energies[method])

        for idx in range(n_plots, rows * cols):
            axes_list[idx].axis("off")

        wind_str = f"Wind: [{wind_vector[0]:.1f}, {wind_vector[1]:.1f}] m/s"
        fig.suptitle(
            f"Drone Route Optimization Comparison ({wind_str})", fontsize=24, y=0.98
        )

        fig.tight_layout(rect=(0, 0.03, 1, 0.95), h_pad=3)
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ---------------- Flexible energy comparison ----------------
    def plot_energy_comparison(
        self,
        results: Dict[str, float],
        filename: str = "total_energy.png",
    ) -> None:
        """
        Saves a bar chart comparing energy costs for multiple solvers.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        display_map = {
            "Naive": "Naive",
            "nearest_neighbor": "NN",
            "nn_2opt": "2-opt",
            "held_karp": "Held-Karp",
            "brute": "Brute",
        }
        labels = [display_map.get(k, k) for k in results.keys()]
        vals = list(results.values())

        cmap = plt.get_cmap("viridis")
        colors = cmap(np.linspace(0.1, 0.9, len(labels)))

        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=1.2)

        ax.set_title(
            "Total Energy Comparison (All Active Solvers)", fontsize=24, pad=15
        )
        ax.set_ylabel("Energy [J]", fontsize=18)
        ax.grid(True, axis="y", alpha=0.35)
        ax.set_axisbelow(True)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:,.0f} J",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )

        ax.set_ylim(0, max(vals) * 1.15)
        fig.tight_layout()
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / filename, dpi=200)
        plt.close(fig)
