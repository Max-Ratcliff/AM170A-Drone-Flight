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
        Saves plots/route_map.png with 3 panels:
          (1) Naive
          (2) Optimized (exact: brute/HK)
          (3) Super (NN + 2-opt)
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))

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
                            lw=2.1,
                            alpha=0.9,
                        ),
                    )

            ax.set_title(title, fontsize=16)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(True, alpha=0.35)
            ax.set_aspect("equal", adjustable="box")

        draw_route(axes[0], naive_order, "Naive Route")
        draw_route(axes[1], optimized_order, "Optimized Route (Exact)")
        draw_route(axes[2], super_order, "Super Route (NN + 2-opt)")

        fig.tight_layout()
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / filename, dpi=150)
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
        Saves plots/total_energy.png with 3 bars:
        Naive, Optimized (exact), Super (NN + 2-opt)
        """
        fig, ax = plt.subplots(figsize=(8.5, 5.8))

        labels = ["Naive", "Optimized", "Super (NN+2opt)"]
        vals = [naive_energy, optimized_energy, super_energy]

        ax.bar(labels, vals)
        ax.set_title("Total Energy Comparison", fontsize=18)
        ax.set_ylabel("Energy [J]", fontsize=13)
        ax.grid(True, axis="y", alpha=0.4)

        fig.tight_layout()
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / filename, dpi=150)
        plt.close(fig)