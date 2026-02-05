import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def get_params(p0, pT, T):
    """
    calucalte and eturn parameters for the velocity model.
    """
    return -6 * (pT - p0) / (T**3)


def drone_dynamics(t, state, a, b, T):
    """
    ODE for drone motion: [vx, vy, ax, ay].
    """
    x, y, vx, vy = state
    ax = a * (2 * t - T)
    ay = b * (2 * t - T)
    return [vx, vy, ax, ay]


def solve_numerical(A, B, T_leg, mass, E_H):
    """
    Solves the trajectory using RK45 and numerical integration.
    """
    a = get_params(A[0], B[0], T_leg)
    b = get_params(A[1], B[1], T_leg)

    sol = solve_ivp(
        drone_dynamics,
        (0, T_leg),
        [A[0], A[1], 0, 0],
        args=(a, b, T_leg),
        t_eval=np.linspace(0, T_leg, 500),
        method="RK45",
    )

    t, (x, y, vx, vy) = sol.t, sol.y
    ax, ay = a * (2 * t - T_leg), b * (2 * t - T_leg)

    power = E_H + np.abs((mass * ax * vx) + (mass * ay * vy))
    e_num = np.trapz(power, t)

    return t, x, y, e_num


def solve_analytical(A, B, T_leg, mass, E_H):
    """
    Computes trajectory and energy using the derived closed-form equations.
    """

    x0, y0 = A
    xT, yT = B
    a = get_params(x0, xT, T_leg)
    b = get_params(y0, yT, T_leg)

    t = np.linspace(0, T_leg, 500)

    # x(t) = a * (t^3/3 - T*t^2/2) + x0
    x_ana = a * (t**3 / 3 - T_leg * t**2 / 2) + x0
    y_ana = b * (t**3 / 3 - T_leg * t**2 / 2) + y0

    dist_sq = (xT - x0) ** 2 + (yT - y0) ** 2
    e_ana = (E_H * T_leg) + (9 / 4) * mass * (dist_sq / T_leg**2)

    return t, x_ana, y_ana, e_ana


def plot_results(t, x_num, y_num, x_ana, y_ana):
    """
    Plots both numerical and analytical results for visual verification.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Component Plot
    ax1.plot(t, x_num, "b-", label="Numerical x(t)", lw=4, alpha=0.3)
    ax1.plot(t, x_ana, "k--", label="Analytical x(t)", lw=1)
    ax1.plot(t, y_num, "r-", label="Numerical y(t)", lw=4, alpha=0.3)
    ax1.plot(t, y_ana, "k:", label="Analytical y(t)", lw=1)
    ax1.set(title="Component Comparison", xlabel="Time (s)", ylabel="Position (m)")
    ax1.legend()
    ax1.grid(True)

    # Parametric Plot
    ax2.plot(x_num, y_num, "g-", label="Numerical Path", lw=4, alpha=0.3)
    ax2.plot(x_ana, y_ana, "k--", label="Analytical Path", lw=1)
    ax2.scatter([x_ana[0], x_ana[-1]], [y_ana[0], y_ana[-1]], color="black", zorder=5)
    ax2.set(
        title="Parametric Path Comparison",
        xlabel="x (m)",
        ylabel="y (m)",
        aspect="equal",
    )
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
    plt.savefig("drone_flight_validation.png", dpi=300)


def main():
    # parameters
    A, B = (0, 0), (10, 20)
    T_total, mass, E_H = 10.0, 1.5, 50.0
    T_leg = T_total / 2

    # get solutions
    t_num, x_num, y_num, e_num = solve_numerical(A, B, T_leg, mass, E_H)
    t_ana, x_ana, y_ana, e_ana = solve_analytical(A, B, T_leg, mass, E_H)

    # Output Results
    print(f"{'Metric':<15} | {'Numerical':<15} | {'Analytical':<15}")
    print("-" * 50)
    # double the engery for round trip
    print(f"{'Energy (J)':<15} | {e_num*2:<15.4f} | {e_ana*2:<15.4f}")
    print(f"{'Final X (m)':<15} | {x_num[-1]:<15.4f} | {x_ana[-1]:<15.4f}")
    print(f"{'Final Y (m)':<15} | {y_num[-1]:<15.4f} | {y_ana[-1]:<15.4f}")
    print(f"\nEnergy Error: {abs(e_num - e_ana) * 2:.2e} J")

    plot_results(t_num, x_num, y_num, x_ana, y_ana)


if __name__ == "__main__":
    main()
