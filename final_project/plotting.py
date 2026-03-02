from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from physics import DronePhysics

PLOTS_DIR = Path(__file__).resolve().parent / "plots"


class Visualizer:
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

    # ---------------- 3-panel route map ----------------
    def plot_routes_three(
        self,
        waypoints: np.ndarray,
        naive_order: list[int],
        optimized_order: list[int],
        super_order: list[int],
        filename: str = "route_map.png",
    ) -> None:
        """
        3-panel route comparison with legend and start/end highlighting.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

        def draw_route(ax: Axes, order: list[int], title: str) -> None:
            # Plot all waypoints
            ax.scatter(
                waypoints[:, 0],
                waypoints[:, 1],
                s=90,
                color="#4C72B0",
                edgecolor="black",
                linewidth=1.2,
                zorder=3,
                label="Waypoint",
            )

            # Highlight start/end node (node 0)
            start_x, start_y = waypoints[0]
            ax.scatter(
                start_x,
                start_y,
                s=200,
                facecolor="none",
                edgecolor="green",
                linewidth=3,
                zorder=4,
                label="Start/End",
            )

            # Label nodes
            for idx, (x, y) in enumerate(waypoints):
                ax.text(x + 10, y + 10, str(idx), fontsize=11)

            # Draw route
            for k in range(len(order)):
                i = order[k]
                j = order[(k + 1) % len(order)]

                xi, yi = waypoints[i]
                xj, yj = waypoints[j]

                ax.plot(
                    [xi, xj],
                    [yi, yj],
                    linestyle="--",
                    linewidth=2,
                    color="blue",
                    alpha=0.9,
                    label="Route" if k == 0 else "",
                )

                # Direction arrows
                ax.annotate(
                    "",
                    xy=(xj, yj),
                    xytext=(xi, yi),
                    arrowprops=dict(
                        arrowstyle="->",
                        lw=1.8,
                        color="blue",
                        alpha=0.8,
                    ),
                )

            ax.set_title(title, fontsize=16)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.grid(True, alpha=0.35)
            ax.set_aspect("equal", adjustable="box")

            # Add legend (avoid duplicates)
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=11)

        draw_route(axes[0], naive_order, "Naive Route")
        draw_route(axes[1], optimized_order, "Nearest Neighbor Route")
        draw_route(axes[2], super_order, "NN + 2-opt Route")

        fig.suptitle("Drone Route Comparison", fontsize=20, y=1.02)

        fig.tight_layout()
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ---------------- 3-bar energy comparison ----------------
    def plot_total_energy_three(
        self,
        naive_energy: float,
        optimized_energy: float,
        super_energy: float,
        filename: str = "total_energy.png",
    ) -> None:
        """
        Saves plots/total_energy.png with 3 colored bars and value labels above.
        """
        fig, ax = plt.subplots(figsize=(9, 6))

        labels = ["Naive", "Nearest Neighbor", "NN + 2-opt"]
        vals = [naive_energy, optimized_energy, super_energy]

        # Nice distinct academic-style colors
        colors = ["#4C72B0", "#55A868", "#C44E52"]

        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=1.2)

        ax.set_title("Total Energy Comparison", fontsize=20, pad=15)
        ax.set_ylabel("Energy [J]", fontsize=14)
        ax.grid(True, axis="y", alpha=0.35)
        ax.set_axisbelow(True)

        # Add value labels above bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:,.0f} J",
                ha="center",
                va="bottom",
                fontsize=12,
            )

        # Add slight vertical padding so text doesn’t clip
        ax.set_ylim(0, max(vals) * 1.15)

        fig.tight_layout()
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / filename, dpi=200)
        plt.close(fig)