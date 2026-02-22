"""Visualization of energy curves and optimal routes."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from physics import DronePhysics

if TYPE_CHECKING:
    pass


class Visualizer:
    """Plots energy vs velocity curves and 2D route maps."""

    def plot_energy_curve(self, physics_model: DronePhysics, distance: float) -> None:
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

        ax.set_xlabel("Velocity [m/s]")
        ax.set_ylabel("Energy [Joules]")
        ax.set_title(f"Segment Energy vs Velocity (d = {distance} m)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("plots/energy_curve.png", dpi=150)
        plt.close(fig)

    def plot_route(
        self,
        waypoints: np.ndarray,
        optimal_order: list[int],
    ) -> None:
        """
        Plot 2D waypoints with arrows in optimal visitation order.

        Args:
            waypoints: N x 2 array of (x, y) coordinates.
            optimal_order: Ordered list of waypoint indices.
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        # Scatter all waypoints
        ax.scatter(
            waypoints[:, 0],
            waypoints[:, 1],
            c="steelblue",
            s=120,
            zorder=3,
            edgecolors="black",
            linewidths=1.5,
        )

        # Draw arrows along optimal path
        for k in range(len(optimal_order)):
            i = optimal_order[k]
            j = optimal_order[(k + 1) % len(optimal_order)]
            xi, yi = waypoints[i, 0], waypoints[i, 1]
            xj, yj = waypoints[j, 0], waypoints[j, 1]
            ax.annotate(
                "",
                xy=(xj, yj),
                xytext=(xi, yi),
                arrowprops=dict(
                    arrowstyle="->",
                    color="darkgreen",
                    lw=1.5,
                ),
            )

        # Label start (index 0) and end
        start_idx = optimal_order[0]
        end_idx = optimal_order[-1]
        ax.scatter(
            waypoints[start_idx, 0],
            waypoints[start_idx, 1],
            marker="o",
            s=200,
            facecolors="none",
            edgecolors="green",
            linewidths=3,
            label="Start",
            zorder=4,
        )
        if end_idx != start_idx:
            ax.scatter(
                waypoints[end_idx, 0],
                waypoints[end_idx, 1],
                marker="s",
                s=200,
                facecolors="none",
                edgecolors="red",
                linewidths=3,
                label="End",
                zorder=4,
            )
        else:
            ax.scatter(
                waypoints[end_idx, 0],
                waypoints[end_idx, 1],
                marker="s",
                s=200,
                facecolors="none",
                edgecolors="red",
                linewidths=3,
                label="End (same as Start)",
                zorder=4,
            )

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("Optimal Drone Route")
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("plots/route_map.png", dpi=150)
        plt.close(fig)
