#!/usr/bin/env python
"""Compare solidus/liquidus melting curves between SPIDER and Zalmoxis.

SPIDER uses Andrault et al. (2011) + Hirschmann (2013) curves stored as
S(P) phase boundaries with separate T(P,S) lookup tables.

Zalmoxis uses Monteux et al. (2016) curves stored directly as T(P),
with a constant 600 K solidus-liquidus offset.

This script converts both to T(P) space and plots the comparison.

Usage
-----
    python compare_melting_curves.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d

SPIDER_ROOT = Path(__file__).resolve().parents[2] / 'SPIDER'
ZALMOXIS_ROOT = Path(__file__).resolve().parents[2] / 'Zalmoxis'
OUTDIR = Path(__file__).resolve().parent


# ── SPIDER lookup table loader ────────────────────────────────


def load_spider_1d(filepath: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Load a SPIDER 1D lookup file (P, S phase boundary).

    Parameters
    ----------
    filepath : Path
        Path to the .dat file.

    Returns
    -------
    P_nd : np.ndarray
        Nondimensional pressure values.
    S_nd : np.ndarray
        Nondimensional entropy values.
    P_scale : float
        Pressure scaling factor [Pa].
    S_scale : float
        Entropy scaling factor [J/kg/K].
    """
    with open(filepath) as f:
        lines = f.readlines()

    # Header: line 0 = "# nheader ndata", line 4 = "# P_scale S_scale"
    scales = lines[4].strip().lstrip('#').split()
    P_scale = float(scales[0])
    S_scale = float(scales[1])

    data = np.loadtxt(filepath, comments='#')
    P_nd = data[:, 0]
    S_nd = data[:, 1]
    return P_nd, S_nd, P_scale, S_scale


def load_spider_2d(filepath: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Load a SPIDER 2D lookup file (P, S) -> T.

    Parameters
    ----------
    filepath : Path
        Path to the temperature .dat file.

    Returns
    -------
    P_nd : np.ndarray
        Nondimensional pressure grid (1D, unique values).
    S_nd : np.ndarray
        Nondimensional entropy grid (1D, unique values).
    T : np.ndarray
        Temperature values [K] (2D: P x S).
    P_scale : float
        Pressure scaling factor [Pa].
    S_scale : float
        Entropy scaling factor [J/kg/K].
    """
    with open(filepath) as f:
        lines = f.readlines()

    # Header: line 0 = "# nheader nP nS"
    dims = lines[0].strip().lstrip('#').split()
    n_P = int(dims[1])
    n_S = int(dims[2])

    # Line 4 = "# P_scale S_scale T_scale"
    scales = lines[4].strip().lstrip('#').split()
    P_scale = float(scales[0])
    S_scale = float(scales[1])
    # T_scale = float(scales[2])  # usually 1.0

    data = np.loadtxt(filepath, comments='#')
    P_nd = data[:, 0]
    S_nd = data[:, 1]
    T = data[:, 2]  # already in K (scale = 1.0)

    # Data: n_S blocks of n_P entries. S is slow (outer), P is fast (inner).
    assert len(data) == n_P * n_S, f'Expected {n_P * n_S} rows, got {len(data)}'

    # Extract 1D grids: P from first block, S from first entry of each block
    P_1d = P_nd[:n_P]  # first n_P entries (P varies within block 0)
    S_1d = S_nd[::n_P][:n_S]  # every n_P-th entry (first entry of each block)

    # Reshape: T[S_index, P_index]
    T_grid = T.reshape(n_S, n_P)

    return P_1d, S_1d, T_grid, P_scale, S_scale


def get_spider_melting_curves() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Extract SPIDER's solidus and liquidus as T(P) curves.

    Returns
    -------
    P_Pa : np.ndarray
        Pressure array [Pa].
    T_solidus : np.ndarray
        Solidus temperature [K].
    T_liquidus : np.ndarray
        Liquidus temperature [K].
    S_solidus : np.ndarray
        Solidus entropy [J/kg/K].
    S_liquidus : np.ndarray
        Liquidus entropy [J/kg/K].
    """
    folder = SPIDER_ROOT / 'lookup_data' / '1TPa-dK09-elec-free'

    # Load phase boundaries: S(P)
    P_sol_nd, S_sol_nd, P_sol_scale, S_sol_scale = load_spider_1d(
        folder / 'solidus_A11_H13.dat'
    )
    P_liq_nd, S_liq_nd, P_liq_scale, S_liq_scale = load_spider_1d(
        folder / 'liquidus_A11_H13.dat'
    )

    # Convert to SI
    P_sol_Pa = P_sol_nd * P_sol_scale
    S_sol_SI = S_sol_nd * S_sol_scale
    P_liq_Pa = P_liq_nd * P_liq_scale
    S_liq_SI = S_liq_nd * S_liq_scale

    # Load T(P, S) tables for solid and melt phases
    P_s_nd, S_s_nd, T_s, Ps_scale, Ss_scale = load_spider_2d(folder / 'temperature_solid.dat')
    P_m_nd, S_m_nd, T_m, Pm_scale, Sm_scale = load_spider_2d(folder / 'temperature_melt.dat')

    # Build interpolators in SI units
    # Grid layout: T_grid[S_index, P_index], so axes are (S, P)
    P_s_SI = P_s_nd * Ps_scale
    S_s_SI = S_s_nd * Ss_scale
    P_m_SI = P_m_nd * Pm_scale
    S_m_SI = S_m_nd * Sm_scale

    # Ensure axes are strictly ascending (S may be negative and ascending)
    T_solid_interp = RegularGridInterpolator(
        (S_s_SI, P_s_SI), T_s, bounds_error=False, fill_value=np.nan
    )
    T_melt_interp = RegularGridInterpolator(
        (S_m_SI, P_m_SI), T_m, bounds_error=False, fill_value=np.nan
    )

    # Evaluate T along the solidus and liquidus
    # Solidus: use solid-phase T(S, P) table at the solidus entropy
    T_solidus = T_solid_interp(np.column_stack([S_sol_SI, P_sol_Pa]))

    # Liquidus: use melt-phase T(S, P) table at the liquidus entropy
    T_liquidus = T_melt_interp(np.column_stack([S_liq_SI, P_liq_Pa]))

    return P_sol_Pa, T_solidus, T_liquidus, S_sol_SI, S_liq_SI


def get_zalmoxis_melting_curves() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Zalmoxis solidus and liquidus T(P) curves.

    Returns
    -------
    P_sol : np.ndarray
        Solidus pressure array [Pa].
    T_sol : np.ndarray
        Solidus temperature [K].
    P_liq : np.ndarray
        Liquidus pressure array [Pa].
    T_liq : np.ndarray
        Liquidus temperature [K].
    """
    folder = ZALMOXIS_ROOT / 'data' / 'melting_curves_Monteux-600'

    sol_data = np.loadtxt(folder / 'solidus.dat', comments='#')
    liq_data = np.loadtxt(folder / 'liquidus.dat', comments='#')

    return sol_data[:, 0], sol_data[:, 1], liq_data[:, 0], liq_data[:, 1]


# ── Plotting ──────────────────────────────────────────────────


def load_spider_1d_phase_boundary(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a SPIDER 1D phase boundary file and return P [Pa], S [J/kg/K].

    Parameters
    ----------
    filepath : Path
        Path to the .dat file.

    Returns
    -------
    P_Pa : np.ndarray
        Pressure [Pa].
    S_SI : np.ndarray
        Entropy [J/kg/K].
    """
    with open(filepath) as f:
        lines = f.readlines()
    scales = lines[4].strip().lstrip('#').split()
    P_scale = float(scales[0])
    S_scale = float(scales[1])
    data = np.loadtxt(filepath, comments='#')
    return data[:, 0] * P_scale, data[:, 1] * S_scale


def get_spider_monteux_as_T(folder: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert the new Monteux S(P) phase boundary files back to T(P).

    Loads solidus_Monteux2016.dat and liquidus_Monteux2016.dat, then
    evaluates T at those (P, S) points using SPIDER's T(P,S) tables.

    Parameters
    ----------
    folder : Path
        SPIDER lookup_data directory.

    Returns
    -------
    P_GPa : np.ndarray
        Common pressure grid [GPa].
    T_solidus : np.ndarray
        Solidus temperature [K].
    T_liquidus : np.ndarray
        Liquidus temperature [K].
    """
    # Load new phase boundary files
    P_sol, S_sol = load_spider_1d_phase_boundary(folder / 'solidus_Monteux2016.dat')
    P_liq, S_liq = load_spider_1d_phase_boundary(folder / 'liquidus_Monteux2016.dat')

    # Load T(P,S) tables — load_spider_2d returns (P_nd, S_nd, T_grid, P_scale, S_scale)
    P_s_nd, S_s_nd, T_s, Ps_scale, Ss_scale = load_spider_2d(folder / 'temperature_solid.dat')
    P_m_nd, S_m_nd, T_m, Pm_scale, Sm_scale = load_spider_2d(folder / 'temperature_melt.dat')

    P_s_SI = P_s_nd * Ps_scale
    S_s_SI = S_s_nd * Ss_scale
    P_m_SI = P_m_nd * Pm_scale
    S_m_SI = S_m_nd * Sm_scale

    interp_solid = RegularGridInterpolator(
        (S_s_SI, P_s_SI),
        T_s,
        bounds_error=False,
        fill_value=np.nan,
    )
    interp_melt = RegularGridInterpolator(
        (S_m_SI, P_m_SI),
        T_m,
        bounds_error=False,
        fill_value=np.nan,
    )

    T_solidus = interp_solid(np.column_stack([S_sol, P_sol]))
    T_liquidus = interp_melt(np.column_stack([S_liq, P_liq]))

    return P_sol / 1e9, P_liq / 1e9, T_solidus, T_liquidus


def plot_comparison():
    """Plot original, Zalmoxis, and homogenized melting curves in T-P space."""
    folder = SPIDER_ROOT / 'lookup_data' / '1TPa-dK09-elec-free'

    # Load original SPIDER A11+H13 curves
    P_sp, T_sol_sp, T_liq_sp, S_sol_sp, S_liq_sp = get_spider_melting_curves()

    # Load Zalmoxis Monteux T(P) curves
    P_sol_zal, T_sol_zal, P_liq_zal, T_liq_zal = get_zalmoxis_melting_curves()

    # Load new homogenized Monteux S(P) -> T(P) round-trip
    P_sol_new_GPa, P_liq_new_GPa, T_sol_new, T_liq_new = get_spider_monteux_as_T(folder)

    # Convert to GPa
    P_sp_GPa = P_sp / 1e9
    P_sol_zal_GPa = P_sol_zal / 1e9
    P_liq_zal_GPa = P_liq_zal / 1e9

    # Mask NaN
    valid_sp = ~np.isnan(T_sol_sp) & ~np.isnan(T_liq_sp)
    valid_new_sol = ~np.isnan(T_sol_new)
    valid_new_liq = ~np.isnan(T_liq_new)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── Panel 1: All curves in T-P space ──────────────────────
    ax = axes[0, 0]
    ax.plot(
        P_sp_GPa[valid_sp],
        T_sol_sp[valid_sp],
        'b-',
        lw=1.5,
        alpha=0.5,
        label='SPIDER solidus (A11+H13, old)',
    )
    ax.plot(
        P_sp_GPa[valid_sp],
        T_liq_sp[valid_sp],
        'r-',
        lw=1.5,
        alpha=0.5,
        label='SPIDER liquidus (A11+H13, old)',
    )
    ax.plot(P_sol_zal_GPa, T_sol_zal, 'b--', lw=2, label='Zalmoxis solidus (Monteux+16)')
    ax.plot(P_liq_zal_GPa, T_liq_zal, 'r--', lw=2, label='Zalmoxis liquidus (Monteux+16)')
    ax.plot(
        P_sol_new_GPa[valid_new_sol],
        T_sol_new[valid_new_sol],
        'b:',
        lw=3,
        label='SPIDER solidus (Monteux, new)',
    )
    ax.plot(
        P_liq_new_GPa[valid_new_liq],
        T_liq_new[valid_new_liq],
        'r:',
        lw=3,
        label='SPIDER liquidus (Monteux, new)',
    )

    ax.axvline(135, color='gray', ls=':', alpha=0.5, label='Earth CMB (~135 GPa)')
    ax.axvline(800, color='gray', ls='--', alpha=0.5, label='~5 M_earth CMB (~800 GPa)')

    ax.set_xlabel('Pressure [GPa]')
    ax.set_ylabel('Temperature [K]')
    ax.set_title('Solidus and Liquidus: All Sources')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_xlim(0, 1000)
    ax.grid(alpha=0.3)

    # ── Panel 2: Round-trip verification ──────────────────────
    ax = axes[0, 1]

    # Interpolate Zalmoxis onto new SPIDER Monteux pressure grid
    f_sol_zal = interp1d(P_sol_zal, T_sol_zal, bounds_error=False, fill_value=np.nan)
    f_liq_zal = interp1d(P_liq_zal, T_liq_zal, bounds_error=False, fill_value=np.nan)

    T_sol_target = f_sol_zal(P_sol_new_GPa[valid_new_sol] * 1e9)
    T_liq_target = f_liq_zal(P_liq_new_GPa[valid_new_liq] * 1e9)

    dT_sol_rt = T_sol_new[valid_new_sol] - T_sol_target
    dT_liq_rt = T_liq_new[valid_new_liq] - T_liq_target

    ax.plot(
        P_sol_new_GPa[valid_new_sol], dT_sol_rt, 'b-', lw=2, label='Solidus round-trip error'
    )
    ax.plot(
        P_liq_new_GPa[valid_new_liq], dT_liq_rt, 'r-', lw=2, label='Liquidus round-trip error'
    )
    ax.axhline(0, color='k', ls='-', lw=0.5)

    ax.set_xlabel('Pressure [GPa]')
    ax.set_ylabel('$\\Delta T$ [K]')
    ax.set_title('Round-Trip Verification (new SPIDER $-$ Zalmoxis target)')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1000)
    ax.set_ylim(-0.1, 0.1)
    ax.grid(alpha=0.3)

    # ── Panel 3: Old vs new SPIDER difference ─────────────────
    ax = axes[1, 0]

    # Interpolate old SPIDER curves onto common grid
    f_sol_old = interp1d(P_sp, T_sol_sp, bounds_error=False, fill_value=np.nan)
    f_liq_old = interp1d(P_sp, T_liq_sp, bounds_error=False, fill_value=np.nan)

    T_sol_old_interp = f_sol_old(P_sol_new_GPa[valid_new_sol] * 1e9)
    T_liq_old_interp = f_liq_old(P_liq_new_GPa[valid_new_liq] * 1e9)

    dT_sol_change = T_sol_new[valid_new_sol] - T_sol_old_interp
    dT_liq_change = T_liq_new[valid_new_liq] - T_liq_old_interp

    ax.plot(
        P_sol_new_GPa[valid_new_sol],
        dT_sol_change,
        'b-',
        lw=2,
        label='Solidus (Monteux $-$ A11+H13)',
    )
    ax.plot(
        P_liq_new_GPa[valid_new_liq],
        dT_liq_change,
        'r-',
        lw=2,
        label='Liquidus (Monteux $-$ A11+H13)',
    )
    ax.axhline(0, color='k', ls='-', lw=0.5)

    ax.set_xlabel('Pressure [GPa]')
    ax.set_ylabel('$\\Delta T$ [K]')
    ax.set_title('Impact of Melting Curve Change on SPIDER')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1000)
    ax.grid(alpha=0.3)

    # ── Panel 4: Melting interval comparison ──────────────────
    ax = axes[1, 1]

    # Compute gaps on a common grid
    P_common = np.arange(0, 714)  # limited by liquidus coverage
    T_sol_new_common = f_sol_zal(P_common * 1e9)
    T_liq_new_common = f_liq_zal(P_common * 1e9)
    gap_monteux = T_liq_new_common - T_sol_new_common

    gap_sp = T_liq_sp[valid_sp] - T_sol_sp[valid_sp]

    ax.plot(P_sp_GPa[valid_sp], gap_sp, 'k-', lw=2, alpha=0.5, label='SPIDER old (A11+H13)')
    ax.plot(
        P_common,
        gap_monteux,
        'k--',
        lw=2,
        label='SPIDER new = Zalmoxis (Monteux+16, const 600 K)',
    )

    ax.set_xlabel('Pressure [GPa]')
    ax.set_ylabel('Liquidus $-$ Solidus [K]')
    ax.set_title('Melting Interval Width')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1000)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    outpath = OUTDIR / 'melting_curves_comparison.pdf'
    fig.savefig(outpath, dpi=150)
    print(f'Saved to {outpath}')
    plt.close(fig)

    # Print summary statistics
    print('\nRound-trip verification:')
    print(
        f'  Solidus |dT|: mean={np.nanmean(np.abs(dT_sol_rt)):.4f} K, '
        f'max={np.nanmax(np.abs(dT_sol_rt)):.4f} K'
    )
    print(
        f'  Liquidus |dT|: mean={np.nanmean(np.abs(dT_liq_rt)):.4f} K, '
        f'max={np.nanmax(np.abs(dT_liq_rt)):.4f} K'
    )
    print('\nImpact on SPIDER:')
    print(
        f'  Solidus change: mean={np.nanmean(dT_sol_change):.0f} K, '
        f'min={np.nanmin(dT_sol_change):.0f} K, max={np.nanmax(dT_sol_change):.0f} K'
    )
    print(
        f'  Liquidus change: mean={np.nanmean(dT_liq_change):.0f} K, '
        f'min={np.nanmin(dT_liq_change):.0f} K, max={np.nanmax(dT_liq_change):.0f} K'
    )
    print('\nCoverage:')
    print(f'  Solidus: {np.sum(valid_new_sol)}/{len(T_sol_new)} points valid')
    print(
        f'  Liquidus: {np.sum(valid_new_liq)}/{len(T_liq_new)} points valid '
        f'(limited by melt T table range)'
    )


if __name__ == '__main__':
    plot_comparison()
