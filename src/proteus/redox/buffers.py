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

        log10 fO2 = 6.54106 − 28163.6 / T + 0.055 · P_GPa / T

    with T in K and **P in GPa**. Returns log10 fO2 in bar.

    Input ``pressure`` is in Pa per the module convention; converted
    internally to GPa. The pressure term is small (0.055 · P / T ≈ 3
    per 120 GPa at 2000 K) and approximates the integrated ΔV of the
    Fe + FeO ⇌ FeO₁.₅ equilibrium.
    """
    P_GPa = pressure / 1e9
    return 6.54106 - 28163.6 / temperature + 0.055 * P_GPa / temperature


def log10_fO2_QFM(temperature: float, pressure: float = 1.0e5) -> float:
    """
    Quartz-Fayalite-Magnetite: 3 Fe₂SiO₄ + O₂ ⇌ 2 Fe₃O₄ + 3 SiO₂.

    Parametrisation (O'Neill 1987; Frost 1991 coefficients):

        log10 fO2 = −25738 / T + 9.00 + 0.092 · P_GPa / T

    with T in K, P in GPa. Input ``pressure`` in Pa is converted
    internally. Returns log10 fO2 in bar.
    """
    P_GPa = pressure / 1e9
    return -25738.0 / temperature + 9.00 + 0.092 * P_GPa / temperature


def log10_fO2_NNO(temperature: float, pressure: float = 1.0e5) -> float:
    """
    Nickel-nickel-oxide: 2 Ni + O₂ ⇌ 2 NiO.

    Parametrisation (O'Neill + Pownceby 1993, Contrib Mineral Petrol
    114, 296-314):

        log10 fO2 = 9.36 − 24930 / T + 0.046 · P_GPa / T

    with T in K, P in GPa. Input ``pressure`` in Pa is converted
    internally. Returns log10 fO2 in bar.
    """
    P_GPa = pressure / 1e9
    return 9.36 - 24930.0 / temperature + 0.046 * P_GPa / temperature


# ------------------------------------------------------------------
# Schaefer+24 Eq 13 — Fe³⁺/Fe²⁺ → log10 fO2 in silicate melt
# ------------------------------------------------------------------
#
# Implementation follows Mariana's formulation (her
# redox_plan/Radially_resolved_fO2.pdf Eq 25), which is the natural-log
# form of Schaefer+24 Eq 13 using the Hirschmann 2022 calibration.
# Mariana's form is:
#
#   ln fO2 = (1/0.196) · [ ln(X_FeO1.5 / X_FeO)
#                         - 1.1492e4 / T
#                         + 6.675
#                         + 2.243 X_Al2O3
#                         + 1.828 X_FeO_T
#                         - 3.201 X_CaO
#                         - 5.854 X_Na2O
#                         - 6.215 X_K2O
#                         + 3.36 (1 - 1673/T - ln(T/1673))
#                         + 7.01e-7 P/T
#                         + 1.54e-10 (T - 1673) P/T
#                         - 3.85e-17 P² / T ]
#   fO2 = exp(ln fO2)
#
# with T in K and P in bar. Note that SiO2, TiO2, MgO, and P2O5 do NOT
# appear in the Mariana formulation — the Hirschmann 2022 calibration
# uses only five oxide couplings (Al2O3, FeO_T, CaO, Na2O, K2O).
#
# Earlier versions of this function (Commits A+B as originally shipped)
# used a different ΔCp/R parametrisation derived from Schaefer Table 4,
# which numerically disagreed with Mariana's Eq 25 by ~6 log units at
# the Earth peridotite anchor. That version was replaced under Commit
# B.5 after round-3 physics review flagged the mismatch.
_MARIANA_A        = 0.196
_MARIANA_C_OVER_T = 1.1492e4     # K
_MARIANA_B        = 6.675        # dimensionless
_MARIANA_AL2O3    = 2.243
_MARIANA_FEO_T    = 1.828
_MARIANA_CAO      = -3.201
_MARIANA_NA2O     = -5.854
_MARIANA_K2O      = -6.215
_MARIANA_DCP      = 3.36         # already in ln form
_MARIANA_T0       = 1673.0       # K
_MARIANA_V1       = 7.01e-7
_MARIANA_V2       = 1.54e-10
_MARIANA_V3       = -3.85e-17


def log10_fO2_schaefer2024(
    X_FeO_liq: float,
    X_FeO1_5: float,
    *,
    temperature: float,
    pressure: float,
    X_SiO2: float = 0.0,      # unused in Mariana Eq 25 but kept in
    X_TiO2: float = 0.0,      #   signature for API stability with
    X_MgO: float = 0.0,       #   Schaefer Table 4 callers that may
    X_P2O5: float = 0.0,      #   compute and pass them.
    X_CaO: float,
    X_Na2O: float,
    X_Al2O3: float,
    X_K2O: float,
) -> float:
    """
    Schaefer+24 Eq 13 via Mariana PDF Eq 25 (Hirschmann 2022 calibration).

    Returns log10 fO2 in bar.

    Parameters
    ----------
    X_FeO_liq, X_FeO1_5 : float
        Mole fractions of FeO and FeO_1.5 in the silicate melt.
    temperature : float
        Melt temperature [K]. Must be positive.
    pressure : float
        Melt pressure in Pa (converted internally to bar for the
        volumetric term; Mariana Eq 25 parameters were calibrated
        with P in bar).
    X_CaO, X_Na2O, X_Al2O3, X_K2O : float
        Mole fractions of these oxides in the melt.
    X_SiO2, X_TiO2, X_MgO, X_P2O5 : float
        Accepted for API stability but not used in Mariana Eq 25.
        Callers that have computed these (e.g. for Schaefer Table 4
        alternative calibrations) may pass them safely; they are
        ignored.

    Notes
    -----
    Domain of validity: silicate melts at magma-ocean conditions,
    T ~ 1200-4000 K, P up to ~120 GPa. At subsolidus conditions, the
    linearised ΔCp term at T₀ = 1673 K becomes inaccurate. The
    dispatcher `log10_fO2_mantle` falls back to
    `stagno2013_peridotite` when `phi_max < phi_crit`.
    """
    if X_FeO_liq <= 0:
        raise ValueError(f'X_FeO_liq must be positive; got {X_FeO_liq}')
    if X_FeO1_5 <= 0:
        raise ValueError(f'X_FeO1_5 must be positive; got {X_FeO1_5}')
    if temperature <= 0:
        raise ValueError(f'temperature must be positive; got {temperature}')
    if pressure < 0:
        raise ValueError(f'pressure must be non-negative; got {pressure}')

    T = float(temperature)
    P_bar = pressure / 1e5  # Pa → bar
    X_FeO_T = X_FeO_liq + X_FeO1_5  # total Fe mole fraction

    ln_ratio = np.log(X_FeO1_5 / X_FeO_liq)
    term_T = -_MARIANA_C_OVER_T / T + _MARIANA_B
    term_oxides = (
        _MARIANA_AL2O3 * X_Al2O3
        + _MARIANA_FEO_T * X_FeO_T
        + _MARIANA_CAO * X_CaO
        + _MARIANA_NA2O * X_Na2O
        + _MARIANA_K2O * X_K2O
    )
    term_cp = _MARIANA_DCP * (1.0 - _MARIANA_T0 / T - np.log(T / _MARIANA_T0))
    term_V = (
        _MARIANA_V1 * P_bar / T
        + _MARIANA_V2 * (T - _MARIANA_T0) * P_bar / T
        + _MARIANA_V3 * P_bar ** 2 / T
    )

    ln_fO2 = (1.0 / _MARIANA_A) * (
        ln_ratio + term_T + term_oxides + term_cp + term_V
    )
    # Convert natural log → log10.
    return float(ln_fO2 / np.log(10))


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
