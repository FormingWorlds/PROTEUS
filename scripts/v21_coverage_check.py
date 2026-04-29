"""Stage 3 coverage check for v2.1 melting-curve consolidation.

Question: does the WB17 (MgSiO3_Wolf_Bower_2018_1TPa) entropy lookup
grid extend to PALEOS-Fei2021 P-T solidus/liquidus temperatures at
all pressures up to ~135 GPa (1 M_E CMB)?

For each P in PALEOS-Fei2021/{solidus,liquidus}_P-T.dat:
  - Look up T_min(P) = temperature_solid(P, S_min) and T_max(P) = temperature_solid(P, S_max)
  - Same for temperature_melt.dat
  - Mark whether PALEOS-Fei2021 T(P) lies inside [T_min, T_max]

If everything is inside the grid, Stage 1 plan stands.
If grid is exceeded somewhere, output the worst-case P and flag it.

Output:
  output_files/v21_coverage_check/coverage.pdf
  output_files/v21_coverage_check/coverage.csv
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FWL_DATA = Path(os.environ.get("FWL_DATA", "/Users/timlichtenberg/work/fwl_data"))
WB17_EOS_DIR = (
    FWL_DATA
    / "interior_lookup_tables"
    / "1TPa-dK09-elec-free"
    / "MgSiO3_Wolf_Bower_2018_1TPa"
)
MELT_ROOT = FWL_DATA / "interior_lookup_tables" / "Melting_curves"

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "output_files" / "v21_coverage_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_grid_TPS(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a 3-column SPIDER P-S table.

    Canonical SPIDER format has 5 header lines. Line 1: `# n_header nP nS`.
    Line 5: `# P_scale S_scale value_scale` (multiply column to recover SI).
    Data starts at line 6.

    Returns P_grid (1D, in SI), S_grid (1D, in SI), value_2D shape (nP, nS).
    """
    with open(path) as fh:
        lines = [fh.readline() for _ in range(5)]
    header1 = lines[0].lstrip("#").split()
    n_header = int(header1[0])
    nP = int(header1[1])
    nS = int(header1[2])
    scales = lines[n_header - 1].lstrip("#").split()
    P_scale = float(scales[0])
    S_scale = float(scales[1])
    val_scale = float(scales[2])

    data = np.loadtxt(path, skiprows=n_header)
    if data.shape[0] != nP * nS:
        raise ValueError(
            f"{path}: header says nP*nS={nP*nS} but file has {data.shape[0]} rows"
        )

    # SPIDER convention: P is inner (fastest varying), S is outer (slowest).
    # Reshape as (nS, nP), then transpose to (nP, nS) for return.
    P_col = data[:, 0].reshape(nS, nP)[0, :] * P_scale
    S_col = data[:, 1].reshape(nS, nP)[:, 0] * S_scale
    val = (data[:, 2].reshape(nS, nP) * val_scale).T  # (nP, nS)
    return P_col, S_col, val


def load_PT(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, comments="#")
    return arr[:, 0], arr[:, 1]


def main() -> None:
    print("Loading WB17 EoS lookup grids ...")
    P_sol, S_sol, T_sol_grid = load_grid_TPS(WB17_EOS_DIR / "temperature_solid.dat")
    P_liq, S_liq, T_liq_grid = load_grid_TPS(WB17_EOS_DIR / "temperature_melt.dat")
    print(f"  temperature_solid.dat: nP={len(P_sol)}, nS={len(S_sol)}")
    print(f"    P range = [{P_sol.min():.3e}, {P_sol.max():.3e}] Pa")
    print(f"    S range = [{S_sol.min():.1f}, {S_sol.max():.1f}] J/kg/K")
    print(f"    T at (P_max, S_min) = {T_sol_grid[-1, 0]:.0f} K")
    print(f"    T at (P_max, S_max) = {T_sol_grid[-1, -1]:.0f} K")
    print(f"  temperature_melt.dat:  nP={len(P_liq)}, nS={len(S_liq)}")
    print(f"    P range = [{P_liq.min():.3e}, {P_liq.max():.3e}] Pa")
    print(f"    S range = [{S_liq.min():.1f}, {S_liq.max():.1f}] J/kg/K")
    print(f"    T at (P_max, S_min) = {T_liq_grid[-1, 0]:.0f} K")
    print(f"    T at (P_max, S_max) = {T_liq_grid[-1, -1]:.0f} K")

    print()
    print("Loading PALEOS-Fei2021 P-T melting curves ...")
    P_sol_paleos, T_sol_paleos = load_PT(MELT_ROOT / "PALEOS-Fei2021/solidus_P-T.dat")
    P_liq_paleos, T_liq_paleos = load_PT(MELT_ROOT / "PALEOS-Fei2021/liquidus_P-T.dat")
    print(f"  solidus: P=[{P_sol_paleos.min():.3e}, {P_sol_paleos.max():.3e}] Pa, "
          f"T=[{T_sol_paleos.min():.0f}, {T_sol_paleos.max():.0f}] K")
    print(f"  liquidus: P=[{P_liq_paleos.min():.3e}, {P_liq_paleos.max():.3e}] Pa, "
          f"T=[{T_liq_paleos.min():.0f}, {T_liq_paleos.max():.0f}] K")

    # For each P in WB17 grid (the CONSUMER coordinate), tabulate
    # T_min(P) = T_grid[i, 0] (lowest S row), T_max(P) = T_grid[i, -1]
    # Then interpolate PALEOS T_sol(P) and T_liq(P) onto the same P axis.
    P_eval = P_sol  # WB17 P grid for solid lookup
    T_solid_min = T_sol_grid[:, 0]
    T_solid_max = T_sol_grid[:, -1]
    T_paleos_sol_at_eval = np.interp(P_eval, P_sol_paleos, T_sol_paleos)

    P_eval_liq = P_liq  # WB17 P grid for melt lookup
    T_liq_min = T_liq_grid[:, 0]
    T_liq_max = T_liq_grid[:, -1]
    T_paleos_liq_at_eval = np.interp(P_eval_liq, P_liq_paleos, T_liq_paleos)

    # Coverage check: at each P, is PALEOS-Fei2021 T inside [T_grid_min, T_grid_max]?
    sol_inside = (T_paleos_sol_at_eval >= T_solid_min) & (T_paleos_sol_at_eval <= T_solid_max)
    liq_inside = (T_paleos_liq_at_eval >= T_liq_min) & (T_paleos_liq_at_eval <= T_liq_max)

    sol_above = T_paleos_sol_at_eval > T_solid_max
    liq_above = T_paleos_liq_at_eval > T_liq_max

    n_total_sol = len(P_eval)
    n_total_liq = len(P_eval_liq)
    n_inside_sol = int(sol_inside.sum())
    n_inside_liq = int(liq_inside.sum())
    n_above_sol = int(sol_above.sum())
    n_above_liq = int(liq_above.sum())

    print()
    print("=" * 72)
    print("COVERAGE SUMMARY")
    print("=" * 72)
    print(
        f"Solidus: {n_inside_sol}/{n_total_sol} P-points inside WB17 grid "
        f"({100*n_inside_sol/n_total_sol:.1f}%); "
        f"{n_above_sol} above grid ceiling"
    )
    print(
        f"Liquidus: {n_inside_liq}/{n_total_liq} P-points inside WB17 grid "
        f"({100*n_inside_liq/n_total_liq:.1f}%); "
        f"{n_above_liq} above grid ceiling"
    )

    # Critical check: the 1 M_E run has CMB pressure ~135 GPa; report
    # coverage at 100, 130, 135 GPa.
    print()
    print("Coverage at key pressures (1 M_E and super-Earth-relevant):")
    print(f"  {'P [GPa]':>8s}  {'PALEOS T_sol':>13s}  {'WB17 T_sol_max':>15s}  "
          f"{'PALEOS T_liq':>13s}  {'WB17 T_liq_max':>15s}  inside?")
    for P_GPa in [10, 30, 50, 100, 130, 135, 140]:
        P_Pa = P_GPa * 1e9
        T_sol_p = float(np.interp(P_Pa, P_sol_paleos, T_sol_paleos))
        T_solmax = float(np.interp(P_Pa, P_sol, T_solid_max))
        T_liq_p = float(np.interp(P_Pa, P_liq_paleos, T_liq_paleos))
        T_liqmax = float(np.interp(P_Pa, P_liq, T_liq_max))
        ok_sol = T_sol_p <= T_solmax
        ok_liq = T_liq_p <= T_liqmax
        ok = "YES" if (ok_sol and ok_liq) else f"NO (sol={ok_sol}, liq={ok_liq})"
        print(f"  {P_GPa:8d}  {T_sol_p:13.0f}  {T_solmax:15.0f}  "
              f"{T_liq_p:13.0f}  {T_liqmax:15.0f}  {ok}")

    # Save CSV
    rows_sol = pd.DataFrame({
        "P_Pa": P_eval,
        "P_GPa": P_eval / 1e9,
        "T_paleos_sol_K": T_paleos_sol_at_eval,
        "wb17_T_solid_min_K": T_solid_min,
        "wb17_T_solid_max_K": T_solid_max,
        "inside_grid": sol_inside,
        "above_grid": sol_above,
    })
    rows_liq = pd.DataFrame({
        "P_Pa": P_eval_liq,
        "P_GPa": P_eval_liq / 1e9,
        "T_paleos_liq_K": T_paleos_liq_at_eval,
        "wb17_T_liq_min_K": T_liq_min,
        "wb17_T_liq_max_K": T_liq_max,
        "inside_grid": liq_inside,
        "above_grid": liq_above,
    })
    rows_sol.to_csv(OUT_DIR / "coverage_solidus.csv", index=False)
    rows_liq.to_csv(OUT_DIR / "coverage_liquidus.csv", index=False)
    print()
    print(f"wrote {OUT_DIR / 'coverage_solidus.csv'}")
    print(f"wrote {OUT_DIR / 'coverage_liquidus.csv'}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=False)
    for ax, P_grid, T_min, T_max, T_paleos, label in [
        (axes[0], P_eval / 1e9, T_solid_min, T_solid_max, T_paleos_sol_at_eval, "(a) solidus"),
        (axes[1], P_eval_liq / 1e9, T_liq_min, T_liq_max, T_paleos_liq_at_eval, "(b) liquidus"),
    ]:
        ax.fill_between(P_grid, T_min, T_max, color="lightblue", alpha=0.5,
                         label="WB17 EoS T grid coverage")
        ax.plot(P_grid, T_min, "b--", lw=0.8, label="WB17 grid floor")
        ax.plot(P_grid, T_max, "b-", lw=0.8, label="WB17 grid ceiling")
        ax.plot(P_grid, T_paleos, "r-", lw=2, label="PALEOS-Fei2021 P-T")
        ax.set_xlabel("Pressure $P$ [GPa]")
        ax.set_ylabel("Temperature $T$ [K]")
        ax.set_title(label)
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 12000)
        ax.axvline(135, color="k", ls=":", lw=0.8, alpha=0.5)
        ax.text(135, 11500, "  1M$_\\oplus$ CMB", fontsize=8, va="top")
        ax.grid(True, alpha=0.3, ls=":")
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "v2.1 coverage check: does WB17 EoS T(P,S) grid contain PALEOS-Fei2021 P-T melting curves?",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_pdf = OUT_DIR / "coverage.pdf"
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    print(f"wrote {out_pdf}")
    plt.close(fig)

    # GO / NO-GO
    print()
    print("=" * 72)
    if n_above_sol == 0 and n_above_liq == 0:
        print("GO: WB17 EoS grid covers PALEOS-Fei2021 melting curves at every P.")
        print("    Stage 1 plan stands as-is.")
    else:
        print("CAVEAT: WB17 EoS grid is exceeded by PALEOS-Fei2021 at some P.")
        print("    Decision tree:")
        print("    - If exceedance is at P < 30 GPa only: clip with warning, proceed.")
        print("    - If exceedance is at P_cmb (~135 GPa): structural problem.")
        # Worst exceedance
        if n_above_sol:
            i = np.argmax(T_paleos_sol_at_eval - T_solid_max)
            P_worst = P_eval[i] / 1e9
            margin = T_paleos_sol_at_eval[i] - T_solid_max[i]
            print(f"    Worst solidus exceedance: P={P_worst:.1f} GPa, "
                  f"T_paleos={T_paleos_sol_at_eval[i]:.0f} K vs grid_max={T_solid_max[i]:.0f} K "
                  f"(margin {margin:+.0f} K)")
        if n_above_liq:
            i = np.argmax(T_paleos_liq_at_eval - T_liq_max)
            P_worst = P_eval_liq[i] / 1e9
            margin = T_paleos_liq_at_eval[i] - T_liq_max[i]
            print(f"    Worst liquidus exceedance: P={P_worst:.1f} GPa, "
                  f"T_paleos={T_paleos_liq_at_eval[i]:.0f} K vs grid_max={T_liq_max[i]:.0f} K "
                  f"(margin {margin:+.0f} K)")
    print("=" * 72)


if __name__ == "__main__":
    main()
