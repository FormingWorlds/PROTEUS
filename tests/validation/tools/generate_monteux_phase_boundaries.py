#!/usr/bin/env python
"""Generate SPIDER-format phase boundary files from Monteux et al. (2016) T(P) curves.

Inverts SPIDER's T(P,S) lookup tables to find S(P) such that:
    T_solid(P, S_solidus) = T_solidus_Monteux(P)
    T_melt(P, S_liquidus) = T_liquidus_Monteux(P)

Writes new solidus/liquidus .dat files in SPIDER's 1D lookup format,
compatible with the existing ``-{phase}_phase_boundary_filename`` options.

Usage
-----
    python generate_monteux_phase_boundaries.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.optimize import brentq

SPIDER_ROOT = Path(__file__).resolve().parents[2] / 'SPIDER'
ZALMOXIS_ROOT = Path(__file__).resolve().parents[2] / 'Zalmoxis'


# ── Loaders ───────────────────────────────────────────────────


def load_spider_2d(filepath: Path) -> dict:
    """Load a SPIDER 2D lookup file (P, S) -> quantity.

    Parameters
    ----------
    filepath : Path
        Path to the .dat file.

    Returns
    -------
    dict
        Keys: P_nd, S_nd, values, P_scale, S_scale, nP, nS.
    """
    with open(filepath) as f:
        lines = f.readlines()

    dims = lines[0].strip().lstrip('#').split()
    nP = int(dims[1])
    nS = int(dims[2])

    scales = lines[4].strip().lstrip('#').split()
    P_scale = float(scales[0])
    S_scale = float(scales[1])

    data = np.loadtxt(filepath, comments='#')
    assert len(data) == nP * nS

    # Layout: S is slow (outer), P is fast (inner)
    P_1d = data[:nP, 0]
    S_1d = data[::nP, 1][:nS]
    values = data[:, 2].reshape(nS, nP)

    return {
        'P_nd': P_1d,
        'S_nd': S_1d,
        'values': values,
        'P_scale': P_scale,
        'S_scale': S_scale,
        'nP': nP,
        'nS': nS,
    }


def load_monteux_curve(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a Zalmoxis Monteux T(P) melting curve.

    Parameters
    ----------
    filepath : Path
        Path to solidus.dat or liquidus.dat.

    Returns
    -------
    P_Pa : np.ndarray
        Pressure [Pa].
    T_K : np.ndarray
        Temperature [K].
    """
    data = np.loadtxt(filepath, comments='#')
    return data[:, 0], data[:, 1]


# ── Inversion ─────────────────────────────────────────────────


def invert_T_to_S(
    T_table: dict,
    P_target_GPa: np.ndarray,
    T_target_func: callable,
) -> np.ndarray:
    """Find S(P) such that T(P, S) = T_target(P) by inverting the lookup table.

    At each pressure P, uses Brent's method to solve T(P, S) - T_target = 0
    for S, where T(P, S) is linearly interpolated from the lookup table.

    Parameters
    ----------
    T_table : dict
        Output of load_spider_2d for a temperature table.
    P_target_GPa : np.ndarray
        Pressure grid [GPa] at which to evaluate.
    T_target_func : callable
        Function P [Pa] -> T [K] for the target melting curve.

    Returns
    -------
    S_result : np.ndarray
        Entropy [J/kg/K] at each pressure. NaN where inversion fails.
    """
    P_SI = T_table['P_nd'] * T_table['P_scale']
    S_SI = T_table['S_nd'] * T_table['S_scale']

    # Build 2D interpolator: T(S, P) in SI units
    interp = RegularGridInterpolator(
        (S_SI, P_SI), T_table['values'],
        bounds_error=False, fill_value=np.nan,
    )

    S_min, S_max = S_SI[0], S_SI[-1]
    if S_min > S_max:
        S_min, S_max = S_max, S_min

    S_result = np.full(len(P_target_GPa), np.nan)

    for i, P_GPa in enumerate(P_target_GPa):
        P_Pa = P_GPa * 1e9
        T_target = T_target_func(P_Pa)

        if np.isnan(T_target):
            continue

        # Define residual: T(S, P) - T_target
        def residual(S):
            T_eval = interp((S, P_Pa))
            if np.isnan(T_eval):
                return np.nan
            return float(T_eval) - T_target

        # Check bracket
        try:
            f_lo = residual(S_min)
            f_hi = residual(S_max)
        except Exception:
            continue

        if np.isnan(f_lo) or np.isnan(f_hi):
            continue

        if f_lo * f_hi > 0:
            # No sign change — target T outside table range at this P
            continue

        try:
            S_sol = brentq(residual, S_min, S_max, xtol=1e-2, rtol=1e-10)
            S_result[i] = S_sol
        except (ValueError, RuntimeError):
            continue

    return S_result


# ── File writer ───────────────────────────────────────────────


def write_spider_phase_boundary(
    filepath: Path,
    P_GPa: np.ndarray,
    S_SI: np.ndarray,
    P_scale: float = 1e9,
    S_scale: float = 4824266.84604467,
):
    """Write a SPIDER-format 1D phase boundary file.

    Parameters
    ----------
    filepath : Path
        Output path.
    P_GPa : np.ndarray
        Pressure grid [GPa].
    S_SI : np.ndarray
        Entropy values [J/kg/K].
    P_scale : float
        Pressure scaling factor [Pa].
    S_scale : float
        Entropy scaling factor [J/kg/K].
    """
    # Filter valid (non-NaN) entries
    valid = ~np.isnan(S_SI)
    P_out = P_GPa[valid]
    S_out = S_SI[valid]

    N = len(P_out)
    P_nd = P_out * 1e9 / P_scale  # Convert GPa -> Pa -> nondim
    S_nd = S_out / S_scale

    with open(filepath, 'w') as f:
        f.write(f'# 5 {N}\n')
        f.write('# Pressure, Entropy, Quantity\n')
        f.write('# column * scaling factor should be SI units\n')
        f.write('# scaling factors (constant) for each column given on line below\n')
        f.write(f'# {P_scale} {S_scale}\n')
        for p, s in zip(P_nd, S_nd):
            f.write(f'{p:.18e} {s:.18e}\n')

    print(f'  Wrote {N} points to {filepath}')


# ── Main ──────────────────────────────────────────────────────


def main():
    folder = SPIDER_ROOT / 'lookup_data' / '1TPa-dK09-elec-free'

    # Load SPIDER T(P,S) tables
    print('Loading SPIDER temperature tables...')
    T_solid = load_spider_2d(folder / 'temperature_solid.dat')
    T_melt = load_spider_2d(folder / 'temperature_melt.dat')

    print(f'  Solid table: {T_solid["nP"]} P x {T_solid["nS"]} S points')
    print(f'  Melt table:  {T_melt["nP"]} P x {T_melt["nS"]} S points')

    P_solid_SI = T_solid['P_nd'] * T_solid['P_scale']
    S_solid_SI = T_solid['S_nd'] * T_solid['S_scale']
    P_melt_SI = T_melt['P_nd'] * T_melt['P_scale']
    S_melt_SI = T_melt['S_nd'] * T_melt['S_scale']

    print(f'  Solid P range: {P_solid_SI[0]/1e9:.1f} - {P_solid_SI[-1]/1e9:.1f} GPa')
    print(f'  Solid S range: {S_solid_SI.min():.1f} - {S_solid_SI.max():.1f} J/kg/K')
    print(f'  Solid T range: {T_solid["values"].min():.1f} - {T_solid["values"].max():.1f} K')
    print(f'  Melt P range:  {P_melt_SI[0]/1e9:.1f} - {P_melt_SI[-1]/1e9:.1f} GPa')
    print(f'  Melt S range:  {S_melt_SI.min():.1f} - {S_melt_SI.max():.1f} J/kg/K')
    print(f'  Melt T range:  {T_melt["values"].min():.1f} - {T_melt["values"].max():.1f} K')

    # Load Monteux T(P) curves
    print('\nLoading Monteux et al. (2016) melting curves...')
    zal_folder = ZALMOXIS_ROOT / 'data' / 'melting_curves_Monteux-600'
    P_sol_Pa, T_sol_K = load_monteux_curve(zal_folder / 'solidus.dat')
    P_liq_Pa, T_liq_K = load_monteux_curve(zal_folder / 'liquidus.dat')

    print(f'  Solidus: {len(P_sol_Pa)} points, '
          f'P = {P_sol_Pa[0]/1e9:.1f} - {P_sol_Pa[-1]/1e9:.1f} GPa, '
          f'T = {T_sol_K[0]:.0f} - {T_sol_K[-1]:.0f} K')
    print(f'  Liquidus: {len(P_liq_Pa)} points, '
          f'P = {P_liq_Pa[0]/1e9:.1f} - {P_liq_Pa[-1]/1e9:.1f} GPa, '
          f'T = {T_liq_K[0]:.0f} - {T_liq_K[-1]:.0f} K')

    # Create interpolation functions
    f_sol = interp1d(P_sol_Pa, T_sol_K, bounds_error=False, fill_value=np.nan)
    f_liq = interp1d(P_liq_Pa, T_liq_K, bounds_error=False, fill_value=np.nan)

    # Use same pressure grid as existing A11_H13 files: 0-999 GPa in 1 GPa steps
    P_grid_GPa = np.arange(0, 1000, 1.0)

    # Invert: find S_solidus(P) from solid T table
    print('\nInverting solid T(P,S) table for Monteux solidus...')
    S_solidus = invert_T_to_S(T_solid, P_grid_GPa, f_sol)
    n_valid_sol = np.sum(~np.isnan(S_solidus))
    print(f'  Valid: {n_valid_sol}/{len(P_grid_GPa)} points')

    # Invert: find S_liquidus(P) from melt T table
    print('Inverting melt T(P,S) table for Monteux liquidus...')
    S_liquidus = invert_T_to_S(T_melt, P_grid_GPa, f_liq)
    n_valid_liq = np.sum(~np.isnan(S_liquidus))
    print(f'  Valid: {n_valid_liq}/{len(P_grid_GPa)} points')

    # Write output files
    outdir = folder
    print(f'\nWriting phase boundary files to {outdir}/')

    write_spider_phase_boundary(
        outdir / 'solidus_Monteux2016.dat',
        P_grid_GPa, S_solidus,
    )
    write_spider_phase_boundary(
        outdir / 'liquidus_Monteux2016.dat',
        P_grid_GPa, S_liquidus,
    )

    # Verification: convert back to T and compare
    print('\nVerification: round-trip T(P) comparison...')
    interp_solid = RegularGridInterpolator(
        (S_solid_SI, P_solid_SI), T_solid['values'],
        bounds_error=False, fill_value=np.nan,
    )
    interp_melt = RegularGridInterpolator(
        (S_melt_SI, P_melt_SI), T_melt['values'],
        bounds_error=False, fill_value=np.nan,
    )

    valid_sol = ~np.isnan(S_solidus)
    valid_liq = ~np.isnan(S_liquidus)

    T_rt_sol = np.array([
        interp_solid((S_solidus[i], P_grid_GPa[i] * 1e9))
        for i in range(len(P_grid_GPa)) if valid_sol[i]
    ])
    T_target_sol = f_sol(P_grid_GPa[valid_sol] * 1e9)

    T_rt_liq = np.array([
        interp_melt((S_liquidus[i], P_grid_GPa[i] * 1e9))
        for i in range(len(P_grid_GPa)) if valid_liq[i]
    ])
    T_target_liq = f_liq(P_grid_GPa[valid_liq] * 1e9)

    dT_sol = np.abs(T_rt_sol - T_target_sol)
    dT_liq = np.abs(T_rt_liq - T_target_liq)

    print(f'  Solidus |dT|: mean={np.mean(dT_sol):.3f} K, max={np.max(dT_sol):.3f} K')
    print(f'  Liquidus |dT|: mean={np.mean(dT_liq):.3f} K, max={np.max(dT_liq):.3f} K')

    # Summary
    print('\nDone. New files:')
    print(f'  {outdir / "solidus_Monteux2016.dat"}')
    print(f'  {outdir / "liquidus_Monteux2016.dat"}')
    print('\nTo use in PROTEUS, update spider.py to reference:')
    print('  liquidus_Monteux2016.dat  (instead of liquidus_A11_H13.dat)')
    print('  solidus_Monteux2016.dat   (instead of solidus_A11_H13.dat)')

    return P_grid_GPa, S_solidus, S_liquidus, valid_sol, valid_liq


if __name__ == '__main__':
    main()
