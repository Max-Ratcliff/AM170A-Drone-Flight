"""Visualization of energy curves (E vs T) and 2D route maps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from physics import DronePhysics

PLOTS_DIR = Path(__file__).resolve().parent / "plots"


class Visualizer:
    """Plots energy vs time and 2D route maps."""

    # -----------------------------------------------------------
    # Energy vs Time (unchanged)
    # -----------------------------------------------------------

    def plot_energy_curve(self, physics_model: DronePhysics, distance: float) -> None:
        T_low, T_high = physics_model.feasible_time_bounds(distance)

        Ts = np.linspace(T_low, T_high, 220)
        Es = np.array([physics_model.segment_energy(distance, T) for T in Ts], dtype=float)

        idx = int(np.argmin(Es))
        T_opt = float(Ts[idx])
        E_opt = float(Es[idx])

        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(Ts, Es, linewidth=2, label="Energy vs Time")

        ax.scatter([T_opt], [E_opt], s=90, zorder=5, label=f"Tmin={T_opt:.2f}s")
        ax.axvline(T_opt, linestyle="--", alpha=0.6)

        ax.set_xlabel("Segment Time T [s]", fontsize=14)
        ax.set_ylabel("Energy [J]", fontsize=14)
        ax.set_title(f"Segment Energy vs Time (d = {distance:.0f} m)", fontsize=18)
        ax.grid(True, alpha=0.5)
        ax.legend(fontsize=11)
        fig.tight_layout()

        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / "energy_curve.png", dpi=150)
        plt.close(fig)

    # -----------------------------------------------------------
    # TWO PANEL ROUTE PLOT (naive + optimized)
    # -----------------------------------------------------------

    def plot_routes(
        self,
        waypoints: np.ndarray,
        naive_order: list[int],
        optimal_order: list[int],
    ) -> None:
        fig, (ax_naive, ax_opt) = plt.subplots(1, 2, figsize=(14, 6))

        def draw_route(ax: Axes, order: list[int], title: str) -> None:
            ax.scatter(waypoints[:, 0], waypoints[:, 1], s=70, zorder=3)

            for idx, (x, y) in enumerate(waypoints):
                ax.text(x + 10, y + 10, str(idx), fontsize=12)

            num_arrows_per_segment = 4
            arrow_style = "->,head_width=0.15,head_length=0.1"

            for k in range(len(order)):
                i = order[k]
                j = order[(k + 1) % len(order)]
                xi, yi = waypoints[i]
                xj, yj = waypoints[j]

                for t in np.linspace(0, 1, num_arrows_per_segment + 1)[:-1]:
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
                            lw=2.2,
                            alpha=0.9,
                        ),
                    )

            ax.set_title(title, fontsize=16)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(True, alpha=0.4)
            ax.set_aspect("equal", adjustable="box")

        draw_route(ax_naive, naive_order, "Naive Route")
        draw_route(ax_opt, optimal_order, "Optimized Route")

        fig.tight_layout()
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / "route_map.png", dpi=150)
        plt.close(fig)

    # -----------------------------------------------------------
    # TOTAL ENERGY SEPARATE FIGURE
    # -----------------------------------------------------------

    def plot_total_energy(
        self,
        naive_energy: float,
        optimized_energy: float,
        filename: str = "total_energy.png",
    ) -> None:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))

        labels = ["Naive", "Optimized"]
        vals = [naive_energy, optimized_energy]

        ax.bar(labels, vals)
        ax.set_title("Total Energy Comparison", fontsize=18)
        ax.set_ylabel("Energy [J]", fontsize=13)
        ax.grid(True, axis="y", alpha=0.4)

        fig.tight_layout()
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / filename, dpi=150)
        plt.close(fig)