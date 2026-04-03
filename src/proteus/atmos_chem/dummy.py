"""Dummy atmospheric chemistry module.

Generates parameterized vertical mixing ratio profiles for ~20 species
without running a kinetics solver. Produces output in the same CSV
format as VULCAN for pipeline compatibility.

Physics (0th order):
- Well-mixed layer: surface VMRs from outgassing, constant up to ~0.1 bar.
- Cold trap: H2O follows Clausius-Clapeyron saturation above the cold trap.
- Photolysis layer: O, OH, H, O2, HCN, NO, C2H2 increase above ~1 mbar
  as parameterized dissociation products of H2O, CO2, N2, CH4.
- VMRs renormalized to sum to 1 at each level.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

# Species produced by the dummy module
_PARENT_SPECIES = ['H2O', 'CO2', 'N2', 'H2', 'CO', 'CH4', 'SO2', 'S2', 'H2S', 'NH3', 'O2']
_PHOTO_PRODUCTS = {
    # product: (parent, yield_fraction, description)
    'O': ('CO2', 5e-3, 'CO2 photolysis'),
    'OH': ('H2O', 2e-3, 'H2O photolysis'),
    'H': ('H2O', 3e-3, 'H2O photolysis'),
    'HCN': ('N2', 1e-4, 'N2+CH4 photochemistry'),
    'NO': ('N2', 5e-4, 'N2+O photochemistry'),
    'C2H2': ('CH4', 2e-4, 'CH4 polymerization'),
}
_ALL_SPECIES = _PARENT_SPECIES + list(_PHOTO_PRODUCTS.keys())

# Clausius-Clapeyron parameters for H2O saturation pressure
_L_H2O = 2.5e6      # latent heat [J/kg]
_RV_H2O = 461.5     # specific gas constant for H2O [J/kg/K]
_PSAT_REF = 611.0    # saturation pressure at 273.15 K [Pa]
_TSAT_REF = 273.15   # reference temperature [K]

# Pressure where photolysis products start appearing [Pa]
_P_PHOTO = 100.0  # ~1 mbar


def _saturation_vmr(T, P_total_Pa):
    """H2O saturation VMR from Clausius-Clapeyron."""
    P_sat = _PSAT_REF * np.exp(_L_H2O / _RV_H2O * (1.0 / _TSAT_REF - 1.0 / np.maximum(T, 100.0)))
    return np.minimum(P_sat / P_total_Pa, 1.0)


def _build_profiles(hf_row, config, num_levels=50):
    """Build parameterized vertical profiles for all species.

    Parameters
    ----------
    hf_row : dict
        Current helpfile row.
    config : Config
        PROTEUS configuration.
    num_levels : int
        Number of vertical levels.

    Returns
    -------
    dict
        Keys: 'tmp', 'p', 'z', 'Kzz', and species names. Values: 1D arrays
        ordered from TOA (index 0) to surface (index -1).
    """
    P_surf = max(float(hf_row.get('P_surf', 1.0)), 1e-3)  # bar
    T_surf = float(hf_row.get('T_surf', hf_row.get('T_magma', 3000.0)))
    gravity = float(hf_row.get('gravity', 9.81))
    p_top = config.atmos_clim.p_top  # bar

    # Pressure grid (log-spaced, TOA to surface)
    p_bar = np.logspace(np.log10(max(p_top, 1e-8)), np.log10(P_surf), num_levels)
    p_Pa = p_bar * 1e5

    # Simple temperature profile: isothermal stratosphere + adiabatic troposphere
    T_strat = max(T_surf * 0.5, 200.0)  # stratospheric temperature
    P_tropo = P_surf * 0.1  # tropopause at 10% of surface pressure
    T = np.where(
        p_bar > P_tropo,
        T_strat + (T_surf - T_strat) * (np.log(p_bar / P_tropo) / np.log(P_surf / P_tropo)),
        T_strat,
    )
    T = np.clip(T, 150.0, T_surf)

    # Altitude from hydrostatic balance (simple scale height integration)
    mmw = float(hf_row.get('atm_kg_per_mol', 0.028))
    H = 8314.0 * T / (mmw * 1e3 * gravity) if gravity > 0 else np.full_like(T, 8000.0)
    z = np.zeros(num_levels)
    for i in range(num_levels - 2, -1, -1):
        dz = H[i] * np.log(p_bar[i + 1] / p_bar[i])
        z[i] = z[i + 1] + abs(dz)

    # Eddy diffusivity: simple power law (decreases with pressure)
    Kzz = 1e8 * (p_bar[-1] / np.maximum(p_bar, 1e-10)) ** 0.5  # cm2/s

    # Surface VMRs from outgassing
    surface_vmr = {}
    for sp in _PARENT_SPECIES:
        surface_vmr[sp] = max(float(hf_row.get(f'{sp}_vmr', 0.0)), 0.0)

    # Build vertical profiles
    profiles = {}

    # Parent species: well-mixed below photolysis level
    for sp in _PARENT_SPECIES:
        vmr = np.full(num_levels, surface_vmr[sp])

        # H2O: apply cold trap (Clausius-Clapeyron)
        if sp == 'H2O' and surface_vmr['H2O'] > 0:
            vmr_sat = _saturation_vmr(T, p_Pa)
            vmr = np.minimum(vmr, vmr_sat)

        profiles[sp] = vmr

    # Photolysis products: increase above P_photo
    for product, (parent, yield_frac, _desc) in _PHOTO_PRODUCTS.items():
        parent_vmr = surface_vmr.get(parent, 0.0)
        if parent_vmr <= 0:
            profiles[product] = np.zeros(num_levels)
            continue

        # Exponential increase toward TOA
        photo_vmr = parent_vmr * yield_frac * np.exp(-p_Pa / _P_PHOTO)
        photo_vmr = np.clip(photo_vmr, 0.0, parent_vmr * 0.1)  # cap at 10% of parent
        profiles[product] = photo_vmr

        # Subtract from parent to conserve atoms (approximate)
        profiles[parent] = np.maximum(profiles[parent] - photo_vmr, 0.0)

    # Normalize VMRs to sum to 1 at each level
    total = np.zeros(num_levels)
    for sp in _ALL_SPECIES:
        total += profiles[sp]

    for sp in _ALL_SPECIES:
        if np.any(total > 0):
            profiles[sp] = np.where(total > 0, profiles[sp] / total, 0.0)

    # Assemble output dict (TOA to surface order)
    result = {
        'tmp': T,
        'p': p_Pa,
        'z': z,
        'Kzz': Kzz,
    }
    for sp in _ALL_SPECIES:
        result[sp] = profiles[sp]

    return result


def run_dummy_chem(
    dirs: dict, config: Config, hf_row: dict, *, online: bool = False
) -> bool:
    """Run dummy atmospheric chemistry and write CSV output.

    Parameters
    ----------
    dirs : dict
        Directory paths.
    config : Config
        PROTEUS configuration.
    hf_row : dict
        Helpfile row (read only).
    online : bool
        If True, write per-snapshot file (dummy_{time}.csv).

    Returns
    -------
    bool
        True if successful.
    """
    outdir = os.path.join(dirs['output'], 'offchem')

    if online:
        os.makedirs(outdir, exist_ok=True)
        csv_name = f'dummy_{int(hf_row["Time"])}.csv'
    else:
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.makedirs(outdir)
        csv_name = 'dummy.csv'

    csv_path = os.path.join(outdir, csv_name)

    # Build profiles
    result = _build_profiles(hf_row, config)

    # Write CSV in VULCAN-compatible format
    header = ''
    arrays = []
    for key in result:
        header += str(key).ljust(14, ' ') + '\t'
        arrays.append(result[key])

    data = np.array(arrays).T
    np.savetxt(csv_path, data, delimiter='\t', fmt='%.8e', header=header, comments='')

    log.info('Dummy chemistry: wrote %d levels x %d species to %s', data.shape[0], len(result) - 4, csv_name)

    return True
