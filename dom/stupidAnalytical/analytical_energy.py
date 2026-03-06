"""
AM170A — "Stupid Analytical" Energy Integration

Attempts closed-form solutions for the drone energy integral WITH drag,
which everyone agrees should just be done numerically.

Three levels of stubbornness:

1. NO DRAG (trivially analytical):
   E = (9/4) * m * d^2 / T^2
   This is the known closed-form. Works perfectly.

2. WITH DRAG, WIND PARALLEL TO FLIGHT (polynomial, analytically solvable):
   When wind is along the flight axis, |v_rel| = |s(t) - w_par|, and the
   power integrand becomes piecewise-polynomial (degree 6). We split at
   the roots and integrate each piece by hand (well, by polynomial arithmetic).

3. WITH DRAG, ARBITRARY 2D WIND (the cursed case):
   |v_rel| = sqrt((s(t) - w_par)^2 + w_perp^2), which puts a sqrt(quartic)
   in the integrand. We throw sympy at it and watch it suffer.

Usage:
    python dom/stupidAnalytical/analytical_energy.py
"""

import sys
from pathlib import Path
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "final_project"))

# =====================================================================
#  SHARED CONSTANTS & NUMERICAL BASELINE
# =====================================================================

RHO = 1.225
CD = 1.0
A_CROSS = 0.1
K_DRAG = 0.5 * RHO * CD * A_CROSS  # 0.06125

MASS = 1.5
E_HOVER = 50.0


def numerical_energy(xA, yA, xB, yB, T_leg, m, wind_x, wind_y, N=2000):
    """The trusty np.trapz baseline — the right way to do this."""
    dx, dy = xB - xA, yB - yA
    ax = 6 * dx / T_leg**3
    ay = 6 * dy / T_leg**3

    t = np.linspace(0, T_leg, N)
    vx = ax * t * (T_leg - t)
    vy = ay * t * (T_leg - t)
    acx = ax * (T_leg - 2 * t)
    acy = ay * (T_leg - 2 * t)

    vrel_x = vx - wind_x
    vrel_y = vy - wind_y
    vrel_mag = np.sqrt(vrel_x**2 + vrel_y**2)

    Fx = m * acx + K_DRAG * vrel_mag * vrel_x
    Fy = m * acy + K_DRAG * vrel_mag * vrel_y

    power = np.abs(Fx * vx + Fy * vy)
    return np.trapz(power, t)


# =====================================================================
#  LEVEL 1: NO DRAG — trivially analytical
# =====================================================================

def analytical_no_drag(xA, yA, xB, yB, T_leg, m):
    """E = (9/4) * m * d^2 / T^2. Done. Go home."""
    d_sq = (xB - xA)**2 + (yB - yA)**2
    return (9.0 / 4.0) * m * d_sq / T_leg**2


# =====================================================================
#  LEVEL 2: DRAG + 1D WIND (parallel to flight direction)
#  The integrand is piecewise polynomial — we can integrate exactly.
# =====================================================================

def _poly_integrate(coeffs, a, b):
    """
    Integrate a polynomial sum(c_i * t^i) from a to b.
    coeffs[i] = coefficient of t^i.
    """
    result = 0.0
    for i, c in enumerate(coeffs):
        power = i + 1
        result += c * (b**power - a**power) / power
    return result


def _poly_multiply(p, q):
    """Multiply two polynomials represented as coefficient lists."""
    n = len(p) + len(q) - 1
    result = [0.0] * n
    for i, ci in enumerate(p):
        for j, cj in enumerate(q):
            result[i + j] += ci * cj
    return result


def _poly_add(p, q):
    """Add two polynomials."""
    n = max(len(p), len(q))
    result = [0.0] * n
    for i in range(len(p)):
        result[i] += p[i]
    for i in range(len(q)):
        result[i] += q[i]
    return result


def _poly_scale(p, s):
    """Scale polynomial by scalar."""
    return [c * s for c in p]


def _poly_negate(p):
    return [-c for c in p]


def analytical_drag_1d_wind(xA, yA, xB, yB, T_leg, m, wind_x, wind_y):
    """
    Analytical energy for drag + wind ONLY when wind is parallel to flight.

    Decomposes into flight-direction coordinates. The relative velocity
    perpendicular to flight must be zero for this to work (w_perp = 0).

    The integrand becomes piecewise polynomial of degree 6, which we
    integrate exactly by splitting at roots of the absolute-value terms.
    """
    dx, dy = xB - xA, yB - yA
    d = np.sqrt(dx**2 + dy**2)
    if d == 0:
        return 0.0

    # unit vector along flight
    ex, ey = dx / d, dy / d

    # wind decomposition
    w_par = wind_x * ex + wind_y * ey      # parallel component
    w_perp = -wind_x * ey + wind_y * ex    # perpendicular component

    if abs(w_perp) > 1e-6:
        return None  # can't do this case with polynomial integration

    T = T_leg
    alpha = 6 * d / T**3  # scalar speed parameter

    # s(t) = alpha * t * (T - t)           speed along flight direction
    # a_s(t) = alpha * (T - 2t)            accel along flight direction
    # v_rel(t) = s(t) - w_par              relative speed (scalar, 1D)
    #
    # Force along flight = m * a_s(t) + k * v_rel * |v_rel|
    # Power = Force * s(t)   (s(t) >= 0 always on [0, T])
    #
    # We need to integrate |Power| over [0, T].

    # Represent everything as polynomials in t:
    # s(t) = alpha*T*t - alpha*t^2 = [0, alpha*T, -alpha]
    s_poly = [0.0, alpha * T, -alpha]

    # a_s(t) = alpha*T - 2*alpha*t = [alpha*T, -2*alpha]
    a_poly = [alpha * T, -2.0 * alpha]

    # v_rel(t) = s(t) - w_par = [-w_par, alpha*T, -alpha]
    vrel_poly = [-w_par, alpha * T, -alpha]

    # v_rel(t) = -alpha*t^2 + alpha*T*t - w_par
    # Roots: alpha*t^2 - alpha*T*t + w_par = 0
    # t = (alpha*T ± sqrt(alpha^2*T^2 - 4*alpha*w_par)) / (2*alpha)
    # t = (T ± sqrt(T^2 - 4*w_par/alpha)) / 2

    # Find roots of v_rel(t) to determine sign intervals
    disc = T**2 - 4 * w_par / alpha
    vrel_roots = []
    if disc > 0:
        sq = np.sqrt(disc)
        r1 = (T - sq) / 2
        r2 = (T + sq) / 2
        if 0 < r1 < T:
            vrel_roots.append(r1)
        if 0 < r2 < T:
            vrel_roots.append(r2)
    elif disc == 0:
        r = T / 2
        if 0 < r < T:
            vrel_roots.append(r)

    # Also find roots of a_s(t) = 0 → t = T/2
    # And s(t) is zero at t=0 and t=T (boundaries), positive in between.

    # Power(t) = [m*a_s(t) + k*v_rel(t)*|v_rel(t)|] * s(t)
    # We need |Power(t)|, which may change sign at roots of:
    #   m*a_s(t) + k*v_rel(t)*|v_rel(t)| = 0
    # This is hard to find analytically in general, so we collect all
    # possible sign-change points and evaluate the sign in each interval.

    # Build the integrand piecewise:
    # In intervals where v_rel > 0: |v_rel| = v_rel
    #   force_poly = m*a_poly + k * vrel_poly * vrel_poly (degree 4)
    #   power_poly = force_poly * s_poly (degree 6)
    # In intervals where v_rel < 0: |v_rel| = -v_rel
    #   force_poly = m*a_poly + k * vrel_poly * (-vrel_poly) = m*a_poly - k*vrel^2
    #   power_poly = force_poly * s_poly

    # vrel^2 as polynomial
    vrel_sq = _poly_multiply(vrel_poly, vrel_poly)

    # force when v_rel > 0: m*a_poly + k*vrel_sq
    force_pos = _poly_add(_poly_scale(a_poly, m), _poly_scale(vrel_sq, K_DRAG))
    power_pos = _poly_multiply(force_pos, s_poly)

    # force when v_rel < 0: m*a_poly - k*vrel_sq
    force_neg = _poly_add(_poly_scale(a_poly, m), _poly_scale(vrel_sq, -K_DRAG))
    power_neg = _poly_multiply(force_neg, s_poly)

    # Collect all critical points where sign of integrand could change
    critical = sorted(set([0.0, T] + vrel_roots + [T / 2.0]))
    # Also numerically find roots of the power polynomial in each interval
    # to handle the absolute value correctly
    all_breaks = [0.0]
    for i in range(len(critical) - 1):
        a_pt, b_pt = critical[i], critical[i + 1]
        mid = (a_pt + b_pt) / 2

        # determine which power polynomial to use based on v_rel sign
        vrel_mid = alpha * mid * (T - mid) - w_par
        poly = power_pos if vrel_mid >= 0 else power_neg

        # find roots of this polynomial in (a_pt, b_pt) numerically
        # (yes, this is a tiny bit numerical, but only for root-finding)
        test_ts = np.linspace(a_pt + 1e-12, b_pt - 1e-12, 200)
        vals = np.array([sum(c * tt**k for k, c in enumerate(poly)) for tt in test_ts])
        sign_changes = np.where(np.diff(np.sign(vals)))[0]
        for idx in sign_changes:
            # bisect to find root
            lo, hi = test_ts[idx], test_ts[idx + 1]
            for _ in range(60):
                mid_r = (lo + hi) / 2
                val = sum(c * mid_r**k for k, c in enumerate(poly))
                if val * sum(c * lo**k for k, c in enumerate(poly)) < 0:
                    hi = mid_r
                else:
                    lo = mid_r
            all_breaks.append((lo + hi) / 2)

    all_breaks.append(T)
    all_breaks = sorted(set(all_breaks))

    # Now integrate |power| over each sub-interval
    total_energy = 0.0
    for i in range(len(all_breaks) - 1):
        a_pt = all_breaks[i]
        b_pt = all_breaks[i + 1]
        mid = (a_pt + b_pt) / 2

        vrel_mid = alpha * mid * (T - mid) - w_par
        poly = power_pos if vrel_mid >= 0 else power_neg

        # evaluate sign of power at midpoint
        val_mid = sum(c * mid**k for k, c in enumerate(poly))

        if val_mid >= 0:
            piece = _poly_integrate(poly, a_pt, b_pt)
        else:
            piece = -_poly_integrate(poly, a_pt, b_pt)

        total_energy += piece

    return total_energy


# =====================================================================
#  LEVEL 3: DRAG + ARBITRARY 2D WIND — sympy symbolic integration
#  This is where we watch the computer suffer.
# =====================================================================

def analytical_drag_2d_sympy(xA, yA, xB, yB, T_leg, m, wind_x, wind_y):
    """
    Attempt to compute the energy integral symbolically using sympy.

    The integrand contains sqrt((polynomial)^2 + constant^2) which
    generally has no elementary antiderivative. We try anyway.

    Returns (result, time_taken) or (None, time_taken) if sympy gives up.
    """
    import sympy as sp

    t = sp.Symbol('t', positive=True)
    T = sp.Rational(T_leg).limit_denominator(1000) if T_leg == int(T_leg) else sp.Float(T_leg)

    dx, dy = xB - xA, yB - yA
    d = np.sqrt(dx**2 + dy**2)
    if d == 0:
        return 0.0, 0.0

    ex, ey = dx / d, dy / d
    alpha = 6 * d / float(T_leg)**3

    # velocity components
    vx = sp.Float(alpha * ex) * t * (sp.Float(T_leg) - t)
    vy = sp.Float(alpha * ey) * t * (sp.Float(T_leg) - t)

    # acceleration components
    acx = sp.Float(alpha * ex) * (sp.Float(T_leg) - 2 * t)
    acy = sp.Float(alpha * ey) * (sp.Float(T_leg) - 2 * t)

    # relative velocity
    vrel_x = vx - sp.Float(wind_x)
    vrel_y = vy - sp.Float(wind_y)
    vrel_mag = sp.sqrt(vrel_x**2 + vrel_y**2)

    # thrust force (compensating for drag)
    k = sp.Float(K_DRAG)
    Fx = sp.Float(m) * acx + k * vrel_mag * vrel_x
    Fy = sp.Float(m) * acy + k * vrel_mag * vrel_y

    # power = |F · v|
    power = sp.Abs(Fx * vx + Fy * vy)

    # THE MOMENT OF TRUTH
    start = time.time()
    try:
        result = sp.integrate(power, (t, 0, sp.Float(T_leg)))
        elapsed = time.time() - start
        # try to evaluate to float
        try:
            val = float(result.evalf())
            return val, elapsed
        except Exception:
            return result, elapsed
    except Exception as e:
        elapsed = time.time() - start
        return None, elapsed


# =====================================================================
#  LEVEL 3b: DRAG + 2D WIND — Gauss-Legendre quadrature
#  Not "analytical" per se, but uses exact polynomial weights rather
#  than brute trapz. With enough points, it's exact for polynomials
#  up to degree 2n-1. Our integrand ISN'T polynomial (that sqrt...),
#  but GL converges exponentially for smooth functions.
# =====================================================================

def gauss_legendre_energy(xA, yA, xB, yB, T_leg, m, wind_x, wind_y, order=40):
    """
    Gauss-Legendre quadrature — the 'fancy numerical' approach that
    converges exponentially for smooth integrands.

    With order=40, this uses 40 carefully chosen points instead of
    trapz's 2000 equally-spaced points, and is generally MORE accurate.
    """
    # GL nodes and weights on [-1, 1]
    nodes, weights = np.polynomial.legendre.leggauss(order)

    # transform to [0, T_leg]
    t_pts = 0.5 * T_leg * (nodes + 1)
    w_pts = 0.5 * T_leg * weights

    dx, dy = xB - xA, yB - yA
    ax = 6 * dx / T_leg**3
    ay = 6 * dy / T_leg**3

    vx = ax * t_pts * (T_leg - t_pts)
    vy = ay * t_pts * (T_leg - t_pts)
    acx = ax * (T_leg - 2 * t_pts)
    acy = ay * (T_leg - 2 * t_pts)

    vrel_x = vx - wind_x
    vrel_y = vy - wind_y
    vrel_mag = np.sqrt(vrel_x**2 + vrel_y**2)

    Fx = m * acx + K_DRAG * vrel_mag * vrel_x
    Fy = m * acy + K_DRAG * vrel_mag * vrel_y

    power = np.abs(Fx * vx + Fy * vy)

    return np.sum(power * w_pts)


# =====================================================================
#  MAIN — RUN ALL APPROACHES AND COMPARE
# =====================================================================

def compare(label, xA, yA, xB, yB, T_leg, m, wx, wy, try_sympy=False):
    """Run all methods for one test case and compare."""
    d = np.sqrt((xB - xA)**2 + (yB - yA)**2)
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  A=({xA},{yA}) → B=({xB},{yB}), d={d:.2f}m, T={T_leg}s")
    print(f"  Wind=({wx}, {wy}) m/s, m={m} kg, k={K_DRAG}")
    print(f"{'='*70}")

    # Numerical baseline
    t0 = time.time()
    e_num = numerical_energy(xA, yA, xB, yB, T_leg, m, wx, wy)
    t_num = time.time() - t0

    # Level 1: no-drag analytical
    e_nodrag = analytical_no_drag(xA, yA, xB, yB, T_leg, m)

    # Level 2: 1D polynomial
    t0 = time.time()
    e_1d = analytical_drag_1d_wind(xA, yA, xB, yB, T_leg, m, wx, wy)
    t_1d = time.time() - t0

    # Level 3b: Gauss-Legendre
    t0 = time.time()
    e_gl = gauss_legendre_energy(xA, yA, xB, yB, T_leg, m, wx, wy, order=40)
    t_gl = time.time() - t0

    print(f"\n  {'Method':<40} {'Energy (J)':>14} {'Time':>12} {'Error vs trapz':>16}")
    print(f"  {'-'*82}")

    print(f"  {'np.trapz (N=2000) [baseline]':<40} {e_num:>14.8f} {t_num*1000:>10.3f}ms {'—':>16}")
    print(f"  {'Analytical no-drag (9/4·m·d²/T²)':<40} {e_nodrag:>14.8f} {'instant':>12} {e_nodrag - e_num:>+16.8f}")

    if e_1d is not None:
        print(f"  {'Polynomial integration (1D wind)':<40} {e_1d:>14.8f} {t_1d*1000:>10.3f}ms {e_1d - e_num:>+16.8f}")
    else:
        print(f"  {'Polynomial integration (1D wind)':<40} {'N/A (w_perp≠0)':>14} {t_1d*1000:>10.3f}ms {'—':>16}")

    print(f"  {'Gauss-Legendre (order=40)':<40} {e_gl:>14.8f} {t_gl*1000:>10.3f}ms {e_gl - e_num:>+16.8f}")

    # Level 3: sympy (only if requested — it's SLOW)
    if try_sympy:
        print(f"\n  Attempting sympy symbolic integration... (this may take a while)")
        e_sym, t_sym = analytical_drag_2d_sympy(xA, yA, xB, yB, T_leg, m, wx, wy)
        if e_sym is not None and isinstance(e_sym, float):
            print(f"  {'SymPy symbolic integration':<40} {e_sym:>14.8f} {t_sym:>10.1f}s  {e_sym - e_num:>+16.8f}")
        elif e_sym is not None:
            print(f"  {'SymPy symbolic integration':<40} {'expr returned':>14} {t_sym:>10.1f}s")
            print(f"    Result: {e_sym}")
        else:
            print(f"  {'SymPy symbolic integration':<40} {'FAILED':>14} {t_sym:>10.1f}s")
            print(f"    sympy could not find a closed-form antiderivative.")
            print(f"    This is because sqrt(quartic) has no elementary integral.")

    return e_num, e_1d, e_gl


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     STUPID ANALYTICAL: Energy Integration Comparison            ║")
    print("║     'Just because you CAN doesn't mean you SHOULD'             ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # ------------------------------------------------------------------
    # Case 1: No wind, no drag baseline
    # ------------------------------------------------------------------
    compare("Case 1: No Wind (drag still present, but wind=0)",
            0, 0, 100, 50, 10.0, MASS, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Case 2: Wind PARALLEL to flight (pure 1D — polynomial solvable)
    # ------------------------------------------------------------------
    compare("Case 2: Wind parallel to flight (+x wind, flight along +x)",
            0, 0, 100, 0, 10.0, MASS, 1.789, 0.0)

    # ------------------------------------------------------------------
    # Case 3: Wind ANTI-PARALLEL (headwind — still 1D polynomial)
    # ------------------------------------------------------------------
    compare("Case 3: Headwind (flight +x, wind -x)",
            0, 0, 100, 0, 10.0, MASS, -1.789, 0.0)

    # ------------------------------------------------------------------
    # Case 4: Wind at 45° to flight (2D — polynomial method fails)
    # ------------------------------------------------------------------
    compare("Case 4: Wind at angle to flight (2D — the cursed case)",
            0, 0, 100, 50, 10.0, MASS, -1.789, 0.5)

    # ------------------------------------------------------------------
    # Case 5: Same as case 4, but with sympy
    # ------------------------------------------------------------------
    print("\n" + "~"*70)
    print("  Now letting sympy try the 2D case...")
    print("  (Timeout yourself if this takes more than 30 seconds)")
    print("~"*70)
    compare("Case 5: 2D wind — sympy attempt",
            0, 0, 100, 50, 10.0, MASS, -1.789, 0.5,
            try_sympy=True)

    # ------------------------------------------------------------------
    # VERDICT
    # ------------------------------------------------------------------
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  VERDICT                                                        ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║                                                                  ║")
    print("║  • No-drag: (9/4)·m·d²/T² works perfectly. Obviously.          ║")
    print("║                                                                  ║")
    print("║  • 1D wind (parallel): Polynomial integration works, but        ║")
    print("║    requires interval splitting at sign changes. More code,       ║")
    print("║    same accuracy, restricted to special geometry.                ║")
    print("║                                                                  ║")
    print("║  • 2D wind (general): sqrt(quartic) in the integrand.           ║")
    print("║    No elementary antiderivative. SymPy either fails or           ║")
    print("║    returns an expression involving elliptic integrals            ║")
    print("║    that still need numerical evaluation.                         ║")
    print("║                                                                  ║")
    print("║  • Gauss-Legendre (order 40) uses 40 points vs trapz's 2000    ║")
    print("║    and gets BETTER accuracy. If you hate trapz, use this.       ║")
    print("║                                                                  ║")
    print("║  CONCLUSION: np.trapz with N=2000 is the right call.            ║")
    print("║  The 'analytical' approaches are either limited (1D only),      ║")
    print("║  impossible (2D), or just numerical in disguise (Gauss-Leg).    ║")
    print("║                                                                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
