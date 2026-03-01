#!/usr/bin/env python
"""Validate SPIDER mesh interpolation from Zalmoxis structure profiles.

Compares the SPIDER mesh file written by write_spider_mesh_file() against
the original Zalmoxis output, verifying that interpolation preserves the
physical profile within acceptable tolerances.

Checks performed:
1. Node count and ordering (surface → CMB, descending radius)
2. Radius range matches Zalmoxis mantle extent
3. Interpolated P, rho, g agree with Zalmoxis at SPIDER node positions
4. Gravity is negative (SPIDER convention)
5. Pressure is monotonically increasing inward
6. Density is positive everywhere
7. No NaN or Inf values

Usage:
    python validate_mesh_interpolation.py <spider_mesh.dat> <zalmoxis_output.dat>

Parameters
----------
spider_mesh.dat : str
    SPIDER mesh file written by write_spider_mesh_file().
    Format: header "# nb ns", then nb+ns lines of "r P rho g" (SI).
zalmoxis_output.dat : str
    Zalmoxis mantle output file.
    Format: 5 columns "r P rho g T" (SI), CMB→surface ascending.

Exit code 0 if all checks pass, 1 otherwise.
"""
from __future__ import annotations

import sys

import numpy as np

RTOL_INTERP = 1e-3  # Relative tolerance for interpolation agreement
RTOL_RANGE = 1e-6   # Tolerance for radius range matching


def load_spider_mesh(filename):
    """Load SPIDER mesh file.

    Returns
    -------
    nb : int
        Number of basic nodes.
    ns : int
        Number of staggered nodes.
    basic : ndarray, shape (nb, 4)
        Columns: r, P, rho, g (SI, surface→CMB).
    staggered : ndarray, shape (ns, 4)
        Columns: r, P, rho, g (SI, surface→CMB).
    """
    with open(filename) as f:
        header = f.readline().strip()
        nb, ns = map(int, header.lstrip("# ").split())
        data = np.loadtxt(f)
    basic = data[:nb]
    staggered = data[nb : nb + ns]
    return nb, ns, basic, staggered


def load_zalmoxis_output(filename):
    """Load Zalmoxis mantle output.

    Returns
    -------
    data : ndarray, shape (N, 5)
        Columns: r, P, rho, g, T (SI, CMB→surface ascending).
    """
    return np.loadtxt(filename)


def check(name, condition, detail=""):
    """Print check result and return pass/fail."""
    status = "PASS" if condition else "FAIL"
    msg = f"  {status} {name}"
    if detail:
        msg += f": {detail}"
    print(msg)
    return condition


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <spider_mesh.dat> <zalmoxis_output.dat>")
        sys.exit(1)

    spider_file = sys.argv[1]
    zalmoxis_file = sys.argv[2]

    # Load data
    nb, ns, basic, staggered = load_spider_mesh(spider_file)
    zal = load_zalmoxis_output(zalmoxis_file)

    r_b = basic[:, 0]
    P_b = basic[:, 1]
    rho_b = basic[:, 2]
    g_b = basic[:, 3]

    r_s = staggered[:, 0]
    P_s = staggered[:, 1]
    rho_s = staggered[:, 2]
    g_s = staggered[:, 3]

    # Zalmoxis columns: r, P, rho, g, T (ascending r: CMB → surface)
    zr = zal[:, 0]
    zP = zal[:, 1]
    zrho = zal[:, 2]
    zg = zal[:, 3]

    print(f"SPIDER mesh: {nb} basic + {ns} staggered nodes")
    print(f"Zalmoxis profile: {len(zal)} points, r=[{zr[0]:.0f}, {zr[-1]:.0f}] m")
    print()

    all_ok = True

    # 1. Node counts
    all_ok &= check("node count", ns == nb - 1,
                     f"nb={nb}, ns={ns}, expected ns={nb - 1}")

    # 2. No NaN/Inf
    all_ok &= check("no NaN in basic", not np.any(np.isnan(basic)))
    all_ok &= check("no Inf in basic", not np.any(np.isinf(basic)))
    all_ok &= check("no NaN in staggered", not np.any(np.isnan(staggered)))

    # 3. Ordering: surface → CMB (descending radius)
    all_ok &= check("basic r descending", np.all(np.diff(r_b) < 0),
                     f"r[0]={r_b[0]:.0f} → r[-1]={r_b[-1]:.0f}")

    if ns > 0:
        all_ok &= check("staggered r descending", np.all(np.diff(r_s) < 0))

    # 4. Radius range matches Zalmoxis mantle extent
    R_surf_spider = r_b[0]
    R_cmb_spider = r_b[-1]
    R_surf_zal = zr[-1]
    R_cmb_zal = zr[0]
    all_ok &= check(
        "R_surface match",
        np.isclose(R_surf_spider, R_surf_zal, rtol=RTOL_RANGE),
        f"SPIDER={R_surf_spider:.0f}, Zalmoxis={R_surf_zal:.0f}",
    )
    all_ok &= check(
        "R_cmb match",
        np.isclose(R_cmb_spider, R_cmb_zal, rtol=RTOL_RANGE),
        f"SPIDER={R_cmb_spider:.0f}, Zalmoxis={R_cmb_zal:.0f}",
    )

    # 5. Gravity is negative (SPIDER convention: inward-pointing)
    all_ok &= check("gravity negative (basic)", np.all(g_b < 0),
                     f"min={g_b.min():.3e}, max={g_b.max():.3e}")
    if ns > 0:
        all_ok &= check("gravity negative (staggered)", np.all(g_s < 0))

    # 6. Pressure monotonically increasing inward (surface → CMB)
    all_ok &= check("P monotonic (basic)", np.all(np.diff(P_b) > 0),
                     f"P[0]={P_b[0]:.3e} → P[-1]={P_b[-1]:.3e}")

    # 7. Density positive everywhere
    all_ok &= check("rho positive (basic)", np.all(rho_b > 0),
                     f"min={rho_b.min():.1f}, max={rho_b.max():.1f}")
    if ns > 0:
        all_ok &= check("rho positive (staggered)", np.all(rho_s > 0))

    # 8. Interpolation accuracy: compare SPIDER mesh against Zalmoxis profile
    # Zalmoxis is ascending r, so interpolate onto SPIDER node positions
    print()
    print("Interpolation accuracy (SPIDER nodes vs Zalmoxis profile):")

    # Zalmoxis gravity is positive; SPIDER is negative
    zg_neg = -np.abs(zg)

    for label, r_nodes, P_nodes, rho_nodes, g_nodes in [
        ("basic", r_b, P_b, rho_b, g_b),
        ("staggered", r_s, P_s, rho_s, g_s),
    ]:
        if len(r_nodes) == 0:
            continue

        # Interpolate Zalmoxis values at SPIDER node positions
        P_expected = np.interp(r_nodes, zr, zP)
        rho_expected = np.interp(r_nodes, zr, zrho)
        g_expected = np.interp(r_nodes, zr, zg_neg)

        for field, actual, expected in [
            ("P", P_nodes, P_expected),
            ("rho", rho_nodes, rho_expected),
            ("g", g_nodes, g_expected),
        ]:
            denom = np.maximum(np.abs(expected), 1e-10)
            rel_err = np.abs(actual - expected) / denom
            max_rel = np.max(rel_err)
            mean_rel = np.mean(rel_err)
            ok = max_rel < RTOL_INTERP
            status = "PASS" if ok else "FAIL"
            print(f"  {status} {label} {field}: max_rel={max_rel:.3e}, "
                  f"mean_rel={mean_rel:.3e}")
            all_ok &= ok

    print()
    if all_ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
