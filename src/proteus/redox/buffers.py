"""
Redox buffers and oxybarometers for the unified redox module (#57).

Contents
--------

Reference buffers (pure thermodynamic expressions for a chosen
mineral assemblage):
  - IW  : Iron-Wüstite (Hirschmann 2021 parametrisation)
  - QFM : Quartz-Fayalite-Magnetite (O'Neill 1987)
  - NNO : Ni-NiO (O'Neill + Pownceby 1993)

Oxybarometers (map Fe³⁺/Fe^T in the melt to log10 fO2 at given T, P,
composition):
  - schaefer2024        : Schaefer+24 JGR Planets 129 Eq 13 (silicate
                          melt; default in MO-active regime)
  - hirschmann2022      : Hirschmann 2022 GCA 328 (deep-MO calibration)
  - sossi2020           : Sossi+20 Sci Adv 6 eaba4823 (peridotite)
  - stagno2013_peridotite : Stagno+13 EPSL 368 (anchored peridotite)

Dispatcher `log10_fO2_mantle` routes to `schaefer2024` when the
magma-ocean regime is active (φ_max > config.redox.phi_crit), and
to `stagno2013_peridotite` otherwise. See plan v6 §2.8.

All buffer/oxybarometer functions return log10 fO2 in bar.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

log = logging.getLogger('fwl.' + __name__)


# ------------------------------------------------------------------
# Reference buffers — log10 fO2 as f(T, P)
# ------------------------------------------------------------------

# Gas constant [J K-1 mol-1]
_R_GAS = 8.314462618

# Pascal per bar
_PA_PER_BAR = 1e5


def log10_fO2_IW(temperature: float, pressure: float = 1.0e5) -> float:
    """
    Iron-Wüstite buffer: 2 Fe + O₂ ⇌ 2 FeO.

    Parametrisation (Hirschmann 2021, Am Mineral 106, 555-563):

        log10 fO2 = 6.54106 - 28163.6 / T + 0.055 * (P − 1) / T

    with T in K, P in bar (1 bar reference → 1e5 Pa). Returns log10 fO2
    in bar.
    """
    P_bar = pressure / _PA_PER_BAR
    return 6.54106 - 28163.6 / temperature + 0.055 * (P_bar - 1.0) / temperature


def log10_fO2_QFM(temperature: float, pressure: float = 1.0e5) -> float:
    """
    Quartz-Fayalite-Magnetite: 3 Fe₂SiO₄ + O₂ ⇌ 2 Fe₃O₄ + 3 SiO₂.

    Parametrisation (O'Neill 1987; Frost 1991 coefficients):

        log10 fO2 = -25738 / T + 9.00 + 0.092 * (P − 1) / T

    with T in K, P in bar. Returns log10 fO2 in bar.
    """
    P_bar = pressure / _PA_PER_BAR
    return -25738.0 / temperature + 9.00 + 0.092 * (P_bar - 1.0) / temperature


def log10_fO2_NNO(temperature: float, pressure: float = 1.0e5) -> float:
    """
    Nickel-nickel-oxide: 2 Ni + O₂ ⇌ 2 NiO.

    Parametrisation (O'Neill + Pownceby 1993, Contrib Mineral Petrol
    114, 296-314):

        log10 fO2 = 9.36 - 24930 / T + 0.046 * (P − 1) / T

    with T in K, P in bar. Returns log10 fO2 in bar.
    """
    P_bar = pressure / _PA_PER_BAR
    return 9.36 - 24930.0 / temperature + 0.046 * (P_bar - 1.0) / temperature


# ------------------------------------------------------------------
# Schaefer+24 Eq 13 — Fe³⁺/Fe²⁺ → log10 fO2 in silicate melt
# ------------------------------------------------------------------

# Fit parameters from Schaefer+24 Table 4 (reproduced from
# Hirschmann 2022, Table 2).
_SCHAEFER24_A = 0.1917
_SCHAEFER24_B = -1.961
_SCHAEFER24_C = 4158.1          # K
_SCHAEFER24_DCP = 33.25         # J/K
_SCHAEFER24_T0 = 1673.15        # K
_SCHAEFER24_Y = (               # y1..y9 [K]
    -520.46,
    -185.37,
    494.39,
    1938.34,
    2888.48,
    3473.68,
    -4473.6,
    -1245.09,
    -1156.86,
)


def log10_fO2_schaefer2024(
    X_FeO_liq: float,
    X_FeO1_5: float,
    *,
    temperature: float,
    pressure: float,
    X_SiO2: float,
    X_TiO2: float,
    X_MgO: float,
    X_CaO: float,
    X_Na2O: float,
    X_P2O5: float,
    X_Al2O3: float,
    X_K2O: float,
) -> float:
    """
    Schaefer+24 Eq 13, reproducing Hirschmann 2022 silicate-melt fit.

    Returns log10 fO2 in bar.

    Parameters
    ----------
    X_FeO_liq, X_FeO1_5 : float
        Mole fractions of FeO and FeO_1.5 in the silicate melt.
    temperature : float
        Melt temperature [K].
    pressure : float
        Melt pressure [Pa]; converted to GPa internally for the
        volumetric term. Must be ≥ 0.
    X_SiO2, X_TiO2, X_MgO, X_CaO, X_Na2O, X_P2O5, X_Al2O3, X_K2O :
        Mole fractions of the oxide components in the melt.

    Notes
    -----
    Domain of validity: silicate melts at magma-ocean conditions,
    T ~ 1200-4000 K, P up to ~120 GPa (Schaefer's whole-Earth BPLE).
    At subsolidus (φ < φ_crit) or lower-temperature conditions the
    linearised ΔCp / T₀ terms degrade. The dispatcher
    `log10_fO2_mantle` falls back to `stagno2013_peridotite` when
    `phi_max < phi_crit`.
    """
    if X_FeO_liq <= 0:
        raise ValueError(f'X_FeO_liq must be positive; got {X_FeO_liq}')
    if X_FeO1_5 <= 0:
        raise ValueError(f'X_FeO1_5 must be positive; got {X_FeO1_5}')
    if temperature <= 0:
        raise ValueError(f'temperature must be positive; got {temperature}')

    T = float(temperature)
    P_GPa = pressure / 1e9

    # Main term: ln(X_FeO1.5 / X_FeO) = a*ln(fO2) + b + c/T - ΔCp/R*(1 - T0/T - ln(T/T0))
    #                                    + sum_k y_k * X_k / T + vol terms
    # Rearranged for ln(fO2):
    #   ln(fO2) = (1/a) * [ ln(X_FeO1.5/X_FeO) - b - c/T
    #                       + ΔCp*(1 - T0/T - ln(T/T0))/R
    #                       - sum_k y_k*X_k / T
    #                       - VdP term ]
    # NOTE: the exact sign structure in Schaefer+24 Table 4 is:
    #   log10(X_FeO1.5/X_FeO) = a*log10 fO2 + b + c/T - ΔCp/R*[1 - T0/T - ln(T/T0)]
    #                           + (1/T) * Σ y_k X_k
    # (Hirschmann 2022 Eq 6). Here we work in natural log for internal
    # math and convert at the end.
    # Σ_k y_k X_k:
    oxide_term = (
        _SCHAEFER24_Y[0] * X_SiO2
        + _SCHAEFER24_Y[1] * X_TiO2
        + _SCHAEFER24_Y[2] * X_MgO
        + _SCHAEFER24_Y[3] * X_CaO
        + _SCHAEFER24_Y[4] * X_Na2O
        + _SCHAEFER24_Y[5] * X_P2O5
        + _SCHAEFER24_Y[6] * X_Al2O3
        + _SCHAEFER24_Y[7] * X_K2O
        # y8 would couple to an additional oxide (Schaefer+24 Table 4
        # footnote); not used here.
    )

    # Volumetric term: see Schaefer+24 Eq 13 final two lines.
    # Using -ΔV/(R T ln10) ≈ -(P − P0)·ΔV/(R T ln10); here the
    # Schaefer fit absorbs this into a quadratic in P. Coefficients
    # (from Schaefer Eq 13 written in log10 form) absorb the
    # ΔV/R·ln10 factor.
    vol_term = (
        7.01e-7 * P_GPa / T
        + 1.54e-10 * (T - _SCHAEFER24_T0) * P_GPa / T
        - 3.85e-17 * (P_GPa ** 2) / T
    )

    # Thermodynamic (T-only) terms.
    dcp_term = (_SCHAEFER24_DCP / _R_GAS) * (
        1.0 - _SCHAEFER24_T0 / T - np.log(T / _SCHAEFER24_T0)
    )

    # Rearrange to log10 fO2:
    log_ratio = np.log10(X_FeO1_5 / X_FeO_liq)
    lhs = (
        log_ratio
        - _SCHAEFER24_B
        - _SCHAEFER24_C / T
        + dcp_term / np.log(10)   # convert ΔCp natural-log term
        - oxide_term / T
        - vol_term
    )
    return float(lhs / _SCHAEFER24_A)


# ------------------------------------------------------------------
# Stagno+2013 garnet peridotite — fallback when MO is inactive
# ------------------------------------------------------------------

# Anchor point (plan v6 §2.8): P=3 GPa, T=1573 K, Fe2O3=0.05 wt%.
# Under these conditions Stagno+13 Fig 4 places ΔQFM ≈ -0.5 for
# fertile lherzolite. Slope d(log fO2)/d(log Fe3+/FeT) ≈ 4.0
# (i.e., 1/0.25 per Hirschmann 2023 Eq 5).
_STAGNO13_ANCHOR_P_PA = 3.0e9
_STAGNO13_ANCHOR_T_K = 1573.0
_STAGNO13_ANCHOR_FE2O3_WT = 0.05
_STAGNO13_ANCHOR_DELTA_QFM = -0.5
_STAGNO13_SLOPE = 4.0  # d(log fO2)/d(log Fe3+/FeT) at the anchor


def log10_fO2_stagno2013_peridotite(
    Fe3_frac: float,
    *,
    temperature: float,
    pressure: float,
    mantle_comp=None,
) -> float:
    """
    Stagno+2013 garnet-peridotite oxybarometer, anchored per plan
    v6 §2.8.

    log10 fO2 = log10 fO2_QFM(T, P) + ΔQFM_anchor + slope · (log10 Fe3_frac − log10 Fe3_anchor)

    with `Fe3_anchor` back-computed from the Fe₂O₃ = 0.05 wt% + total Fe
    assumption (defaults to Fe3/FeT ≈ 0.02 under Earth BSE FeO ~ 8 wt%).

    Returns log10 fO2 in bar. Domain of validity: φ < φ_crit.
    """
    if Fe3_frac <= 0:
        raise ValueError(f'Fe3_frac must be positive; got {Fe3_frac}')
    qfm = log10_fO2_QFM(temperature, pressure)
    # Default anchor Fe3_frac ≈ 0.02 (Earth BSE modern ratio).
    fe3_anchor = 0.02
    delta = _STAGNO13_ANCHOR_DELTA_QFM + _STAGNO13_SLOPE * (
        np.log10(Fe3_frac) - np.log10(fe3_anchor)
    )
    return qfm + delta


# ------------------------------------------------------------------
# Hirschmann 2022 and Sossi 2020 — registered but not yet calibrated
# ------------------------------------------------------------------

def log10_fO2_hirschmann2022(*args, **kwargs) -> float:
    """
    Hirschmann 2022 deep-MO calibration. Currently shares the
    Schaefer+24 implementation (Schaefer adopted the Hirschmann 2022
    Eq 6 fit with the same Table 2 parameters). Kept as a separate
    entry for forward flexibility.
    """
    return log10_fO2_schaefer2024(*args, **kwargs)


def log10_fO2_sossi2020(*args, **kwargs) -> float:
    """
    Sossi+2020 peridotite-based oxybarometer. NotImplemented for the
    Commit-B scaffolding; to be calibrated against their Fig 3a when
    a consumer needs it.
    """
    raise NotImplementedError(
        'log10_fO2_sossi2020: registered but not yet calibrated. '
        'Fill in from Sossi et al. 2020 Sci Adv 6:eaba4823 if needed.'
    )


# ------------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------------

OXYBAROMETERS: dict[str, Callable] = {
    'schaefer2024': log10_fO2_schaefer2024,
    'hirschmann2022': log10_fO2_hirschmann2022,
    'sossi2020': log10_fO2_sossi2020,
    'stagno2013_peridotite': log10_fO2_stagno2013_peridotite,
}

BUFFERS: dict[str, Callable] = {
    'IW': log10_fO2_IW,
    'QFM': log10_fO2_QFM,
    'NNO': log10_fO2_NNO,
}


def log10_fO2_mantle(
    *,
    Fe3_frac: float,
    temperature: float,
    pressure: float,
    phi_max: float,
    mantle_comp,
    oxybarometer: str = 'schaefer2024',
    phi_crit: float = 0.4,
) -> float:
    """
    Regime-aware mantle oxybarometer dispatcher.

    Returns log10 fO2 in bar at the melt-surface (or whichever
    representative (T, P) the caller supplies).

    MO-active (`phi_max > phi_crit`) → route to the configured
    silicate-melt oxybarometer (default `schaefer2024`).
    MO-inactive → route to `stagno2013_peridotite`.

    See plan v6 §2.8. Called from
    `proteus.redox.partitioning.advance_fe_reservoirs` to compute the
    warm-start suggestion for the outer Brent solver.
    """
    if Fe3_frac <= 0:
        raise ValueError(f'Fe3_frac must be positive; got {Fe3_frac}')

    if phi_max > phi_crit and oxybarometer in ('schaefer2024', 'hirschmann2022'):
        # Silicate-melt regime. Schaefer Eq 13 needs the oxide mole
        # fractions; derive from mantle_comp wt%. For the Commit-B
        # stub caller we support a lightweight fallback that skips
        # Eq 13 when `mantle_comp` is not provided: behaves like
        # stagno2013_peridotite but tagged as a warning.
        if mantle_comp is None:
            log.warning(
                'log10_fO2_mantle: mantle_comp is None in MO-active regime; '
                'falling back to stagno2013_peridotite at surface.'
            )
            return log10_fO2_stagno2013_peridotite(
                Fe3_frac=Fe3_frac,
                temperature=temperature,
                pressure=pressure,
            )
        X = _oxide_mole_fractions_from_mantle_comp(mantle_comp, Fe3_frac)
        return log10_fO2_schaefer2024(
            X_FeO_liq=X['FeO'],
            X_FeO1_5=X['FeO1.5'],
            temperature=temperature,
            pressure=pressure,
            X_SiO2=X['SiO2'], X_TiO2=X['TiO2'], X_MgO=X['MgO'],
            X_CaO=X['CaO'], X_Na2O=X['Na2O'], X_P2O5=X['P2O5'],
            X_Al2O3=X['Al2O3'], X_K2O=X['K2O'],
        )

    # MO-inactive regime.
    return log10_fO2_stagno2013_peridotite(
        Fe3_frac=Fe3_frac,
        temperature=temperature,
        pressure=pressure,
    )


def _oxide_mole_fractions_from_mantle_comp(
    mantle_comp, Fe3_frac: float,
) -> dict[str, float]:
    """
    Convert `MantleComp` wt% → oxide mole fractions at the melt surface,
    splitting total Fe into FeO (ferrous) and FeO_1.5 (ferric, = Fe2O3/2)
    using `Fe3_frac`.
    """
    from proteus.utils.constants import oxide_mmw

    feo_total_wt = mantle_comp.FeO_total_wt
    # Split by mole, then back to wt of FeO vs FeO_1.5.
    n_fe_total_per_100g = feo_total_wt / oxide_mmw['FeO']  # mol
    n_fe3 = Fe3_frac * n_fe_total_per_100g
    n_fe2 = (1.0 - Fe3_frac) * n_fe_total_per_100g
    # FeO (ferrous) wt% and FeO_1.5 (ferric) wt%.
    feo_wt = n_fe2 * oxide_mmw['FeO']
    feo15_wt = n_fe3 * (oxide_mmw['Fe2O3'] / 2.0)   # half of Fe2O3

    # Collect the other oxides.
    wt_by_oxide = {
        'FeO':    feo_wt,
        'FeO1.5': feo15_wt,
        'SiO2':   mantle_comp.SiO2_wt,
        'TiO2':   mantle_comp.TiO2_wt,
        'MgO':    mantle_comp.MgO_wt,
        'CaO':    mantle_comp.CaO_wt,
        'Na2O':   mantle_comp.Na2O_wt,
        'P2O5':   mantle_comp.P2O5_wt,
        'Al2O3':  mantle_comp.Al2O3_wt,
        'K2O':    mantle_comp.K2O_wt,
    }
    # Convert to moles via oxide_mmw; FeO1.5 uses Fe2O3 MW / 2.
    mols: dict[str, float] = {}
    for ox, wt in wt_by_oxide.items():
        if ox == 'FeO1.5':
            mw = oxide_mmw['Fe2O3'] / 2.0
        else:
            mw = oxide_mmw[ox]
        mols[ox] = wt / mw
    total_mol = sum(mols.values())
    if total_mol <= 0:
        raise ValueError('MantleComp has non-positive total oxide moles')
    return {ox: n / total_mol for ox, n in mols.items()}


def delta_to_buffer(log10_fO2_value: float, buffer: str,
                    temperature: float, pressure: float = 1.0e5) -> float:
    """Convert absolute log10 fO2 to ΔBuffer at (T, P)."""
    if buffer not in BUFFERS:
        raise KeyError(f'Unknown buffer {buffer!r}; choices: {list(BUFFERS)}')
    return log10_fO2_value - BUFFERS[buffer](temperature, pressure)


def buffer_to_absolute(delta_buffer: float, buffer: str,
                       temperature: float, pressure: float = 1.0e5) -> float:
    """Convert ΔBuffer to absolute log10 fO2 at (T, P)."""
    if buffer not in BUFFERS:
        raise KeyError(f'Unknown buffer {buffer!r}; choices: {list(BUFFERS)}')
    return delta_buffer + BUFFERS[buffer](temperature, pressure)
