"""Visualization of energy curves (E vs T) and 2D route maps."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from physics import DronePhysics

if TYPE_CHECKING:
    pass

PLOTS_DIR = Path(__file__).resolve().parent / "plots"


class Visualizer:
    """Plots energy vs time and 2D route maps."""

    def plot_energy_curve(self, physics_model: DronePhysics, distance: float) -> None:
        """
        Plot U-shaped Energy vs Time curve for a segment of fixed distance,
        and mark Tmin.
        """
        T_low, T_high = physics_model.feasible_time_bounds(distance)

        Ts = np.linspace(T_low, T_high, 220)
        Es = np.array([physics_model.segment_energy(distance, T) for T in Ts], dtype=float)

        idx = int(np.argmin(Es))
        T_opt = float(Ts[idx])
        E_opt = float(Es[idx])

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(Ts, Es, linewidth=2, label="Energy vs Time")

        ax.scatter([T_opt], [E_opt], s=90, zorder=5, label=f"Tmin={T_opt:.2f}s")
        ax.axvline(T_opt, linestyle="--", alpha=0.6)

        ax.set_xlabel("Segment Time T [s]", fontsize=14)
        ax.set_ylabel("Energy [J]", fontsize=14)
        ax.set_title(f"Segment Energy vs Time (d = {distance:.0f} m)", fontsize=16)
        ax.grid(True, alpha=0.6)
        ax.legend(fontsize=11)
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
        Plot route map and energy comparison bars.
        """
        fig, (ax_route, ax_bars) = plt.subplots(1, 2, figsize=(14, 6))

        num_arrows_per_segment = 4
        arrow_style = "->,head_width=0.15,head_length=0.1"

        def draw_route(ax: Axes, order: list[int], linestyle: str, lw: float) -> None:
            for k in range(len(order)):
                i = order[k]
                j = order[(k + 1) % len(order)]
                xi, yi = waypoints[i, 0], waypoints[i, 1]
                xj, yj = waypoints[j, 0], waypoints[j, 1]

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
                            linestyle=linestyle,
                            lw=lw,
                            alpha=0.75 if linestyle == "--" else 1.0,
                        ),
                    )

        # route panel
        ax_route.scatter(waypoints[:, 0], waypoints[:, 1], s=65, zorder=3)
        for idx, (x, y) in enumerate(waypoints):
            ax_route.text(x + 10, y + 10, str(idx), fontsize=12)

        draw_route(ax_route, naive_order, linestyle="--", lw=1.8)
        draw_route(ax_route, optimal_order, linestyle="-", lw=2.4)

        ax_route.set_title("Routes (dashed = naive, solid = optimized)")
        ax_route.set_xlabel("x")
        ax_route.set_ylabel("y")
        ax_route.grid(True, alpha=0.4)

        # bar panel
        labels = ["Naive", "Optimized"]
        vals = [naive_energy, optimal_energy]
        ax_bars.bar(labels, vals)
        ax_bars.set_title("Total Energy Comparison")
        ax_bars.set_ylabel("Energy [J]")
        ax_bars.grid(True, axis="y", alpha=0.4)

        fig.tight_layout()
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / "route_map.png", dpi=150)
        plt.close(fig)