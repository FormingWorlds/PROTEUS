"""
Melt redox evolution (Fe3+ / Fe2+) through magma-ocean crystallization, and
the resulting radially resolved oxygen fugacity, per the accompanying paper
draft ("Time evolution of radially resolved fO2 mantle of rocky exoplanets").

Implements Steps 0-10 of the redox derivation, stepped through the full
SPIDER time series in DATA_DIR:

  Step 0   fixed parameters (f_0, mu_FeO, w_FeT, D_fe2_brg, BSE wt%). D_Fe3+
           is depth-dependent: D_fe3_brg below P_CUTOFF_GPA = 22 GPa
           (bridgmanite), D_fe3_shallow above it (Cpx/Opx, Schaefer et al.
           2024 Table 3 and Eq 6), since pressure_s is time-invariant per
           cell this split is assigned once from the first snapshot.
  Step 1   initial total iron inventory in the melt                  (Eq 1)
  Step 2   initial Fe3+ / Fe2+ reservoirs                             (Eq 2-3)
  Step 3   new solid mass formed each step from Delta(phi)            (Eq 4-6)
  Step 4   redistribute global melt reservoirs across cells           (Eq 9-11,
           done once at t=0 to seed the loop, then again after every
           step using the just-updated global reservoirs so the next
           iteration's Step 5-6 has a "(t-1)" cell state to work from)
  Step 5-6 Fe3+ / Fe2+ partitioning into the newly formed solid        (Eq 12-17)
  Step 7   update global melt reservoirs                              (Eq 18-19)
  Step 8-9 global ferric fraction and Fe3+/Fe2+ redox ratio            (Eq 20-24)
  Step 10  radial fO2 profile from T(r) and the (single, global,
           time-evolving) redox ratio, via Hirschmann (2022) GCA 313
           Eq 21 / Table 2 (= Schaefer et al. 2024's Eq 13 / Table 4)

Note: Eq 9-11 redistribute the melt composition in proportion to local melt
mass, which makes the Fe3+/Fe2+ *ratio* identical in every melt-bearing cell
at a given timestep -- only the absolute Fe3+/Fe2+ amounts vary by cell, in
proportion to local melt mass. So the radial *structure* seen in the fO2
profile below comes entirely from T(r), not from any radial variation in
the ratio itself (there is none, by construction).

Step 10's pressure/EOS term (the integral of Delta V dP) is fixed at zero,
matching every place Schaefer's own code evaluates this equation
(fO2lowP_H22.m, valid only near 1 bar) -- checked directly against their
MATLAB source, which never evaluates a nonzero Delta V dP for this equation
anywhere (the Deng et al. 2020 EOS machinery they do carry, in
deltaGFeOFeO15_Deng.m/BM4VolumeFunc.m, is used only for a different
reaction, the Fe-metal disproportionation equilibrium, not for fO2 itself).
So the radial profile here varies only through T(r); it does not capture
the pressure-driven Fe3+ stabilization that motivates this whole redox
story (Hirschmann 2012; Armstrong et al. 2019; Deng et al. 2020) -- adding
that would require porting the Deng EOS for a purpose Schaefer's own code
does not use it for.

The surface point of that same profile (evaluated at the true surface T, P
from the _b boundary grid, not cell 0 of the _s grid which sits at ~1-2 GPa)
is additionally converted to Delta-IW using the Hirschmann (2021) GCA 313,
74-84 iron-wustite buffer (IW_H21.m in the Schaefer+2024 archive), valid
~100 kPa to 100 GPa.

Grid convention:
  _s  = staggered nodes (99 cells): pressure, mass, radius, phi_s, temp_s
  _b  = basic/boundary nodes (100 cells): phi_b, temperature, etc.

Cells are ordered top -> bottom (index 0 = surface, index 98 = CMB).
Values in the JSON are stored as strings; multiply by the 'scaling' field
to get physical units (e.g. radius_s * scaling = metres).

Checked against the SPIDER output used here: mass_s, pressure_s and radius_s
are exactly time-invariant per cell (0.0% difference between the first and
last available snapshot), so the fixed radial/pressure grid from the first
snapshot is reused for all plots..
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "output/data"

# ── Step 0: fixed input parameters ────────────────────────────────────────────
w_FeT      = 0.08       # bulk iron mass fraction in the melt (dimensionless)
mu_FeO     = 0.07184    # molar mass of FeO in kg/mol (= 71.84 g/mol)
f_0        = 0.10       # initial ferric fraction Fe3+/FeT (Hirschmann 2022)
D_fe2_brg  = 0.85       # bridgmanite/melt partition coefficient for Fe2+ (both regimes)

# depth-dependent D_Fe3+: bridgmanite below P_CUTOFF_GPA (deep, high-pressure
# MgSiO3 polymorph), Cpx/Opx above it (shallow, low-pressure MgSiO3 polymorphs)
P_CUTOFF_GPA = 22.0     # Schaefer et al. (2024) Sec 2.5.1: halt bridgmanite
                        # assemblage above this pressure
D_fe3_brg  = 0.75       # bridgmanite/melt (Schaefer et al. 2024, Table 3)
D_fe3_cpx  = 0.45       # clinopyroxene/melt (Table 3, Mallmann & O'Neill 2009)
D_fe3_opx  = 0.70 * D_fe3_cpx   # orthopyroxene/melt = D_opx/cpx x D_cpx/melt (Eq 6)
# no modal Cpx/Opx proportions given, so -- as a first approximation, in the
# same spirit as using one D_fe3_brg for the whole deep phase -- take the
# plain mean of Cpx and Opx as the single shallow-regime D_Fe3+
D_fe3_shallow = (D_fe3_cpx + D_fe3_opx) / 2

# BSE starting composition (wt%) -- McDonough (2003) via Hirschmann (2022)
wt_FeO   = 7.82
wt_MgO   = 38.3
wt_SiO2  = 45.5
wt_CaO   = 3.58
wt_Al2O3 = 4.49
wt_FeO15 = 0.36   # FeO1.5
wt_Na2O  = 0.0    # not listed in BSE table
wt_K2O   = 0.0    # not listed in BSE table
wt_TiO2  = 0.0    # not listed in BSE table (real BSE value ~0.20 wt%)
wt_P2O5  = 0.0    # not listed in BSE table (real BSE value ~0.02 wt%)

# how many leading crystallization steps get the full Step 1-10 printout
VERBOSE_STEPS = 1

# per-cell, per-timestep dump of every quantity from Steps 3-10
CSV_PATH = "redox_full_timeseries.csv"
CSV_HEADER = [
    "step", "time_years", "cell",
    "radius_m", "pressure_Pa", "temp_K",
    "phi_prev", "phi_curr", "delta_phi", "new_solid_frac", "new_solid_mass_kg",
    "M_melt_cell_kg", "n_fe3_melt_cell_mol", "n_fe2_melt_cell_mol",
    "D_fe3_used", "delta_n_fe3_solid_mol", "delta_n_fe2_solid_mol",
    "n_fe3_melt_global_mol", "n_fe2_melt_global_mol", "n_FeT_melt_global_mol",
    "ferric_frac_global", "redox_ratio_global",
    "log10_fO2_cell",
    "log10_fO2_surf", "dIW_surface",
]
CSV_INT_COLS = (0, 2)   # "step" and "cell" -- everything else is float
# per-column width = max(header length, formatted-value length) + 1 space of
# margin, so columns are only as wide as they need to be; purely cosmetic
_CSV_INT_VALUE_WIDTH   = 6    # e.g. "   307"
_CSV_FLOAT_VALUE_WIDTH = 13   # e.g. "-2.753865e+00"
CSV_COL_WIDTHS = [
    max(len(name), _CSV_INT_VALUE_WIDTH if i in CSV_INT_COLS else _CSV_FLOAT_VALUE_WIDTH) + 1
    for i, name in enumerate(CSV_HEADER)
]


def _csv_format_header(header):
    """Right-justify header labels to CSV_COL_WIDTHS so columns line up."""
    return [f"{name:>{CSV_COL_WIDTHS[i]}}" for i, name in enumerate(header)]


def _csv_format_row(values):
    """Right-justify each value to its column's width, purely for readability
    when the CSV is opened in a plain-text/monospace viewer."""
    return [f"{int(v):>{CSV_COL_WIDTHS[i]}d}" if i in CSV_INT_COLS
            else f"{float(v):>{CSV_COL_WIDTHS[i]}.6e}"
            for i, v in enumerate(values)]


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_field(data, key):
    """Return (values_as_floats, scaling_factor) for one field in a JSON dict."""
    entry   = data["data"][key]
    values  = [float(v) for v in entry["values"]]
    scaling = float(entry["scaling"])
    return values, scaling


def read_spider_step(filepath):
    """
    Load one SPIDER JSON file and return physical profiles as a dict.

    Keys:
        time_years  - simulation time in years
        step        - SPIDER step index
        phi         - melt fraction per cell (dimensionless)
        pressure    - pressure per cell (Pa)
        mass        - mass per cell (kg)
        radius      - radius per cell (m)
        temp        - temperature per cell (K)
    """
    with open(filepath) as f:
        raw = json.load(f)

    phi,      _          = _load_field(raw, "phi_s")
    pressure, P_scale    = _load_field(raw, "pressure_s")
    mass,     mass_scale = _load_field(raw, "mass_s")
    radius,   r_scale    = _load_field(raw, "radius_s")
    temp,     T_scale    = _load_field(raw, "temp_s")

    # true surface (top _b boundary node, P ~ 0), distinct from cell 0 of the
    # _s grid above, which is already offset to ~1-2 GPa
    pressure_b, Pb_scale = _load_field(raw, "pressure_b")
    temp_b,     Tb_scale = _load_field(raw, "temp_b")

    return {
        "time_years":    raw["time_years"],
        "step":          raw["step"],
        "phi":           phi,
        "pressure":      [p * P_scale    for p in pressure],
        "mass":          [m * mass_scale for m in mass],
        "radius":        [r * r_scale    for r in radius],
        "temp":          [t * T_scale    for t in temp],
        "pressure_surf": pressure_b[0] * Pb_scale,
        "temp_surf":     temp_b[0] * Tb_scale,
    }


def compute_hirschmann_mole_fractions(wt_FeO, wt_FeO15, wt_MgO, wt_SiO2, wt_CaO,
                                       wt_Al2O3, wt_Na2O, wt_K2O, wt_TiO2, wt_P2O5):
    """
    Single-cation-basis oxide mole fractions for Hirschmann (2022) Eq 21:
    Na2O, K2O, Al2O3 and P2O5 have 2 cations per formula unit and are
    expressed per single cation ("NaO0.5", "KO0.5", "AlO1.5", "PO2.5"),
    matching Hirschmann's point (2) modification to Eq 20. FeO/FeO15 are
    already single-Fe-cation formulas. All mole fractions are normalized
    over every cation present (including Fe), matching the order used in
    fO2lowP_H22.m: SiO2, TiO2, MgO, CaO, NaO0.5, KO0.5, PO2.5, AlO1.5.
    """
    M = {"SiO2": 60.08, "TiO2": 79.87, "MgO": 40.30, "CaO": 56.08,
         "Na2O": 61.98, "K2O": 94.20, "P2O5": 141.94, "Al2O3": 101.96,
         "FeO": 71.84, "FeO15": 79.84}
    cations_per_unit = {"SiO2": 1, "TiO2": 1, "MgO": 1, "CaO": 1,
                        "Na2O": 2, "K2O": 2, "P2O5": 2, "Al2O3": 2,
                        "FeO": 1, "FeO15": 1}
    wt = {"SiO2": wt_SiO2, "TiO2": wt_TiO2, "MgO": wt_MgO, "CaO": wt_CaO,
          "Na2O": wt_Na2O, "K2O": wt_K2O, "P2O5": wt_P2O5, "Al2O3": wt_Al2O3,
          "FeO": wt_FeO, "FeO15": wt_FeO15}
    n_cation = {k: cations_per_unit[k] * wt[k] / M[k] for k in M}
    n_total  = sum(n_cation.values())
    return {k: v / n_total for k, v in n_cation.items()}


# ── Step 1 (Eq 1) ──────────────────────────────────────────────────────────────

def init_iron_inventory(snapshot, w_FeT, mu_FeO):
    """
    Total iron inventory in the melt at t=0 (Eq 1).

    Returns M_melt_0 (kg), m_FeT_melt_0 (kg), n_FeT_melt_0 (mol).
    """
    n = len(snapshot["phi"])
    M_melt_0     = sum(snapshot["phi"][c] * snapshot["mass"][c] for c in range(n))
    m_FeT_melt_0 = w_FeT * M_melt_0
    n_FeT_melt_0 = m_FeT_melt_0 / mu_FeO

    return {
        "M_melt_0":     M_melt_0,
        "m_FeT_melt_0": m_FeT_melt_0,
        "n_FeT_melt_0": n_FeT_melt_0,
    }


# ── Step 2 (Eq 2-3) ─────────────────────────────────────────────────────────────

def split_iron_reservoirs(iron, f_0):
    """Split total melt iron into Fe3+ and Fe2+ using f_0 (Eq 2-3)."""
    n_FeT = iron["n_FeT_melt_0"]
    return {
        "n_fe3_melt_0": f_0 * n_FeT,
        "n_fe2_melt_0": (1 - f_0) * n_FeT,
    }


# ── Step 3 (Eq 4-6) ─────────────────────────────────────────────────────────────

def compute_new_solid(step_prev, step_curr):
    """
    Newly formed solid between two consecutive timesteps (Eq 4-6).

    delta_phi      - decrease in melt fraction (negative = remelting)
    new_solid_frac - fraction of cell mass that solidified this step (>= 0)
    new_solid_mass - mass of new solid in kg (>= 0)
    """
    n = len(step_prev["phi"])

    delta_phi      = [step_prev["phi"][c] - step_curr["phi"][c]  for c in range(n)]
    new_solid_frac = [max(0.0, d)                                  for d in delta_phi]
    new_solid_mass = [step_curr["mass"][c] * new_solid_frac[c]     for c in range(n)]

    return {
        "delta_phi":      delta_phi,
        "new_solid_frac": new_solid_frac,
        "new_solid_mass": new_solid_mass,
    }


# ── Step 4 (Eq 9-11) ────────────────────────────────────────────────────────────

def distribute_melt_composition(snapshot, n_fe3_melt, n_fe2_melt):
    """
    Redistribute global melt Fe3+ / Fe2+ across cells in proportion to local
    melt mass (Eq 9-11). The output of this call at timestep t is used as the
    "(t-1)" cell state feeding Step 5-6 (Eq 12-17) at the *next* timestep.
    """
    n = len(snapshot["phi"])

    M_melt_cell = [snapshot["phi"][c] * snapshot["mass"][c] for c in range(n)]  # Eq 8
    M_melt_all  = sum(M_melt_cell)                                              # Eq 9

    if M_melt_all > 0:
        n_fe3_melt_cell = [(M_melt_cell[c] / M_melt_all) * n_fe3_melt for c in range(n)]  # Eq 10
        n_fe2_melt_cell = [(M_melt_cell[c] / M_melt_all) * n_fe2_melt for c in range(n)]   # Eq 11
    else:
        # numerical safeguard, not in Eq 9-11: no melt left anywhere to redistribute
        n_fe3_melt_cell = [0.0] * n
        n_fe2_melt_cell = [0.0] * n

    return {
        "M_melt_cell":     M_melt_cell,
        "M_melt_all":      M_melt_all,
        "n_fe3_melt_cell": n_fe3_melt_cell,
        "n_fe2_melt_cell": n_fe2_melt_cell,
    }


# ── Step 5 (Eq 12-14) ───────────────────────────────────────────────────────────

def compute_fe3_partitioning(melt_dist, solid, D_fe3_cell):
    """
    Fe3+ moles entering the newly formed solid per cell (Eq 12-14).

    D_fe3_cell is a per-cell partition coefficient (not a single scalar, as
    in the original Eq 14): D_fe3_brg below P_CUTOFF_GPA (bridgmanite),
    D_fe3_shallow above it (Cpx/Opx) -- see P_CUTOFF_GPA and D_fe3_shallow.
    """
    n = len(melt_dist["n_fe3_melt_cell"])
    delta_n_fe3_solid = []

    for c in range(n):
        M_melt_i = melt_dist["M_melt_cell"][c]
        if M_melt_i > 0:
            C_fe3_melt_i  = melt_dist["n_fe3_melt_cell"][c] / M_melt_i             # Eq 12
            delta_n_fe3_i = D_fe3_cell[c] * C_fe3_melt_i * solid["new_solid_mass"][c]  # Eq 14
        else:
            delta_n_fe3_i = 0.0
        delta_n_fe3_solid.append(delta_n_fe3_i)

    return {"delta_n_fe3_solid": delta_n_fe3_solid}


# ── Step 6 (Eq 15-17) ───────────────────────────────────────────────────────────

def compute_fe2_partitioning(melt_dist, solid, D_fe2_brg):
    """Fe2+ moles entering the newly formed solid per cell (Eq 15-17)."""
    n = len(melt_dist["n_fe2_melt_cell"])
    delta_n_fe2_solid = []

    for c in range(n):
        M_melt_i = melt_dist["M_melt_cell"][c]
        if M_melt_i > 0:
            C_fe2_melt_i  = melt_dist["n_fe2_melt_cell"][c] / M_melt_i             # Eq 15
            delta_n_fe2_i = D_fe2_brg * C_fe2_melt_i * solid["new_solid_mass"][c]  # Eq 17
        else:
            delta_n_fe2_i = 0.0
        delta_n_fe2_solid.append(delta_n_fe2_i)

    return {"delta_n_fe2_solid": delta_n_fe2_solid}


# ── Step 7 (Eq 18-19) ───────────────────────────────────────────────────────────

def update_melt_reservoirs(n_fe3_melt_prev, n_fe2_melt_prev, fe3_part, fe2_part):
    """Update global melt Fe3+ / Fe2+ reservoirs after crystallization (Eq 18-19)."""
    n_fe3_melt = n_fe3_melt_prev - sum(fe3_part["delta_n_fe3_solid"])  # Eq 18
    n_fe2_melt = n_fe2_melt_prev - sum(fe2_part["delta_n_fe2_solid"])  # Eq 19

    return {
        "n_fe3_melt": n_fe3_melt,
        "n_fe2_melt": n_fe2_melt,
    }


# ── Step 8-9 (Eq 20-24) ─────────────────────────────────────────────────────────

def compute_ferric_fraction(melt_update):
    """Updated total iron and ferric fraction of the melt (Eq 20-21)."""
    n_fe3 = melt_update["n_fe3_melt"]
    n_fe2 = melt_update["n_fe2_melt"]

    n_FeT_melt  = n_fe3 + n_fe2           # Eq 20
    ferric_frac = n_fe3 / n_FeT_melt      # Eq 21

    return {
        "n_FeT_melt":  n_FeT_melt,
        "ferric_frac": ferric_frac,
    }


def compute_redox_ratio(ferric):
    """Fe3+/Fe2+ mole fraction ratio from the ferric fraction (Eq 22-24)."""
    f = ferric["ferric_frac"]
    return {"redox_ratio": f / (1.0 - f)}


# ── Step 10 (Hirschmann 2022 Eq 21 = Schaefer 2024 Eq 13): radial fO2 profile ───

def compute_fO2(redox_ratio_val, snapshot, X):
    """
    fO2 per cell using Hirschmann (2022) GCA 313 Eq 21 / Table 2 (Schaefer
    et al. 2024's Eq 13 / Table 4), evaluated at each cell's local T(r) with
    the single, global, time-evolving redox ratio. Ported from fO2lowP_H22.m.

    The pressure/EOS term (integral of Delta V dP) is fixed at zero, as it
    is everywhere Schaefer's own code evaluates this equation -- see module
    docstring. So this has no P-dependence; radial structure comes only
    from T(r).
    """
    R       = 8.31447
    a       = 0.19317
    b       = -4.51412 / 2.303
    c_param = 9574.293 / 2.303
    deltaCp = 33.25
    T0      = 1673.15
    ys = [y / 2.303 for y in
          (-1198.4, -426.82, 1138.371, 4232.933, 6650.972, 7998.434,
           -10298.6, -2866.92, -2663.74)]

    Xi = [X["SiO2"], X["TiO2"], X["MgO"], X["CaO"],
          X["Na2O"], X["K2O"], X["P2O5"], X["Al2O3"]]
    loggammas_const = (
        ys[0]*Xi[0] + ys[1]*Xi[1] + ys[2]*Xi[2] + ys[3]*Xi[3]
        + ys[4]*Xi[4] + ys[5]*Xi[5] + ys[6]*Xi[6]
        + ys[7]*Xi[7]*Xi[0] + ys[8]*Xi[0]*Xi[2]
    )
    logXFe3Fe2 = math.log10(redox_ratio_val)

    ln_fO2_list = []
    fO2_list    = []

    for cell in range(len(snapshot["temp"])):
        T = snapshot["temp"][cell]
        dG_RT       = b + c_param / T - (deltaCp / R / math.log(10)) * (1 - T0 / T - math.log(T / T0))
        loggammas   = loggammas_const / T
        log10_fO2_c = (logXFe3Fe2 - dG_RT - loggammas) / a
        ln_fO2_c    = log10_fO2_c * math.log(10)
        ln_fO2_list.append(ln_fO2_c)
        fO2_list.append(math.exp(ln_fO2_c))

    return {"ln_fO2": ln_fO2_list, "fO2": fO2_list}


def iw_buffer_hirschmann2021(T, P_GPa):
    """
    log10(fO2) of the iron-wustite (IW) buffer, Hirschmann (2021) GCA 313,
    74-84, Table 1, ported from IW_H21.m. Valid ~100 kPa to 100 GPa,
    1000-3000 K. T in K, P in GPa.
    """
    P_transition = -18.640 + 0.04359 * T - 5.069e-6 * T**2

    if P_GPa < P_transition:
        a_coeff = (6.844864, 1.175691e-1, 1.143873e-3, 0, 0)
        b_coeff = (5.791364e-4, -2.891434e-4, -2.737171e-7, 0, 0)
        c_coeff = (-7.971469e-5, 3.198005e-5, 0, 1.059554e-10, 2.014461e-7)
        d_coeff = (-2.769002e4, 5.285977e2, -2.919275e0, 0, 0)
    else:
        a_coeff = (8.463095, -3.000307e-3, 7.213445e-5, 0, 0)
        b_coeff = (1.148738e-3, -9.352312e-5, 5.161592e-7, 0, 0)
        c_coeff = (-7.448624e-4, -6.329325e-6, 0, -1.407339e-10, 1.830014e-4)
        d_coeff = (-2.782082e4, 5.285977e2, -8.473231e-1, 0, 0)

    P_terms = (1, P_GPa, P_GPa**2, P_GPa**3, math.sqrt(P_GPa))
    a_val = sum(a * p for a, p in zip(a_coeff, P_terms))
    b_val = sum(b * p for b, p in zip(b_coeff, P_terms))
    c_val = sum(c * p for c, p in zip(c_coeff, P_terms))
    d_val = sum(d * p for d, p in zip(d_coeff, P_terms))

    return a_val + b_val * T + c_val * T * math.log(T) + d_val / T


# ── verbose diagnostic printout for the first N steps ─────────────────────────

def _print_verbose_step(step_no, iron, fe_split, melt_dist, solid,
                         fe3_part, fe2_part, melt_update,
                         ferric_frac, redox_ratio_val, prev, curr,
                         log10_fO2_cell, log10_fO2_surf, dIW_surface):
    n = len(prev["phi"])

    print()
    print(f"═══ crystallization step {step_no}  "
          f"(SPIDER step {prev['step']} -> {curr['step']}, "
          f"t = {prev['time_years']:.1f} -> {curr['time_years']:.1f} yr) ═══")

    melt_mass = [curr["phi"][c] * curr["mass"][c] for c in range(n)]

    print(f"{'cell':>5}  {'phi_prev':>9}  {'phi_curr':>9}  {'delta_phi':>10}  "
          f"{'new_solid_frac':>15}  {'new_solid_mass (kg)':>20}  {'melt_mass (kg)':>16}  "
          f"{'d_fe3_solid':>12}  {'d_fe2_solid':>12}")
    print("-" * 145)
    for c in range(n):
        print(f"{c:>5}  {prev['phi'][c]:>9.4f}  {curr['phi'][c]:>9.4f}  "
              f"{solid['delta_phi'][c]:>10.4f}  {solid['new_solid_frac'][c]:>15.4f}  "
              f"{solid['new_solid_mass'][c]:>20.3e}  {melt_mass[c]:>16.3e}  "
              f"{fe3_part['delta_n_fe3_solid'][c]:>12.3e}  {fe2_part['delta_n_fe2_solid'][c]:>12.3e}")

    print()
    print(f"Cells with new solid : {sum(1 for m in solid['new_solid_mass'] if m > 0)}")
    print(f"Total new solid mass : {sum(solid['new_solid_mass']):.3e} kg")

    print()
    print("── Step 1: initial iron inventory ───────────────────────")
    print(f"  M_melt_0     = {iron['M_melt_0']:.4e} kg")
    print(f"  m_FeT_melt_0 = {iron['m_FeT_melt_0']:.4e} kg")
    print(f"  n_FeT_melt_0 = {iron['n_FeT_melt_0']:.4e} mol")

    print()
    print("── Step 2: initial Fe3+ / Fe2+ reservoirs ───────────────")
    print(f"  f_0           = {f_0}")
    print(f"  n_fe3_melt_0  = {fe_split['n_fe3_melt_0']:.4e} mol")
    print(f"  n_fe2_melt_0  = {fe_split['n_fe2_melt_0']:.4e} mol")

    print()
    print("── Step 4: melt composition distributed across cells ────")
    print(f"  M_melt_all              = {melt_dist['M_melt_all']:.4e} kg")
    print(f"  sum n_fe3_melt_cell     = {sum(melt_dist['n_fe3_melt_cell']):.4e} mol")
    print(f"  sum n_fe2_melt_cell     = {sum(melt_dist['n_fe2_melt_cell']):.4e} mol")

    print()
    print("── Step 5-6: Fe3+ / Fe2+ partitioning into new solid ────")
    print(f"  D_fe3: {D_fe3_brg} (Brg, P>={P_CUTOFF_GPA} GPa) / {D_fe3_shallow:.4f} (Cpx+Opx, P<{P_CUTOFF_GPA} GPa)"
          f"   total delta_n_fe3_solid = {sum(fe3_part['delta_n_fe3_solid']):.4e} mol")
    print(f"  D_fe2_brg = {D_fe2_brg} (both regimes)"
          f"   total delta_n_fe2_solid = {sum(fe2_part['delta_n_fe2_solid']):.4e} mol")

    print()
    print("── Step 7: updated global melt reservoirs ───────────────")
    print(f"  n_fe3_melt (curr) = {melt_update['n_fe3_melt']:.4e} mol")
    print(f"  n_fe2_melt (curr) = {melt_update['n_fe2_melt']:.4e} mol")

    print()
    print("── Step 8-9: ferric fraction and redox ratio ────────────")
    print(f"  ferric_frac = {ferric_frac:.6f}  (initial f_0 = {f_0})")
    print(f"  redox_ratio = {redox_ratio_val:.6f}")

    print()
    print("── Step 10: radial fO2 profile (Hirschmann 2022 Eq 21, ΔVdP=0) ──────────")
    print(f"  log10(fO2) — top cell (c=0)   : {log10_fO2_cell[0]:.3f}")
    print(f"  log10(fO2) — mid cell (c={n//2}) : {log10_fO2_cell[n//2]:.3f}")
    print(f"  log10(fO2) — bot cell (c={n-1}) : {log10_fO2_cell[n-1]:.3f}")
    print(f"  log10(fO2) — min / max        : {min(log10_fO2_cell):.3f} / {max(log10_fO2_cell):.3f}")
    print(f"  log10(fO2) at true surface (T={curr['temp_surf']:.1f} K, P~0) : {log10_fO2_surf:.3f}")
    print(f"  dIW at true surface                                : {dIW_surface:+.3f}")
    print()


# ── time-stepping driver ───────────────────────────────────────────────────────

def run_redox_evolution(files, verbose_steps=VERBOSE_STEPS, csv_path=CSV_PATH):
    """
    Step through the full SPIDER time series, evolving the global melt
    Fe3+/Fe2+ reservoirs (Steps 1-9) and computing the radial fO2 profile
    plus the surface fO2/dIW (Step 10) at every timestep. Also writes one
    row per cell per timestep to csv_path, covering every quantity from
    Steps 3-10 (Eq 4-26).

    Returns (results, snap0) where results holds per-timestep time series and
    snap0 is the first snapshot (used for the fixed pressure/radius grid).
    """
    snapshots = [read_spider_step(f) for f in files]
    n_cells = len(snapshots[0]["phi"])
    X = compute_hirschmann_mole_fractions(wt_FeO, wt_FeO15, wt_MgO, wt_SiO2, wt_CaO,
                                           wt_Al2O3, wt_Na2O, wt_K2O, wt_TiO2, wt_P2O5)

    # pressure_s is time-invariant per cell (checked against the SPIDER
    # output, see module docstring), so the bridgmanite/Cpx+Opx cutoff can be
    # assigned once from the first snapshot rather than every timestep
    D_fe3_cell = [D_fe3_brg if snapshots[0]["pressure"][c] >= P_CUTOFF_GPA * 1e9
                  else D_fe3_shallow
                  for c in range(n_cells)]

    iron     = init_iron_inventory(snapshots[0], w_FeT, mu_FeO)
    fe_split = split_iron_reservoirs(iron, f_0)
    n_fe3_melt = fe_split["n_fe3_melt_0"]
    n_fe2_melt = fe_split["n_fe2_melt_0"]

    # Step 4 at t=0: seed the "(t-1)" cell reservoirs for the first iteration
    melt_dist = distribute_melt_composition(snapshots[0], n_fe3_melt, n_fe2_melt)

    ferric_frac     = f_0
    redox_ratio_val = f_0 / (1.0 - f_0)
    melt_exhausted  = False   # true once no melt remains anywhere to track

    results = {
        "time_years":     [],
        "step":           [],
        "ferric_frac":    [],
        "redox_ratio":    [],
        "log10_fO2_cell": [],   # per-cell radial profile (Hirschmann 2022 Eq 21)
        "log10_fO2_surf": [],
        "dIW_surface":    [],
    }

    print(f"{'step':>7}  {'time (yr)':>14}  {'ferric_frac':>12}  {'redox_ratio':>12}  "
          f"{'log10fO2 min':>13}  {'log10fO2 max':>13}  {'dIW surf':>10}")
    print("-" * 92)

    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(_csv_format_header(CSV_HEADER))

    for i in range(1, len(snapshots)):
        prev, curr = snapshots[i - 1], snapshots[i]

        solid = compute_new_solid(prev, curr)

        if not melt_exhausted and melt_dist["M_melt_all"] > 0:
            fe3_part = compute_fe3_partitioning(melt_dist, solid, D_fe3_cell)
            fe2_part = compute_fe2_partitioning(melt_dist, solid, D_fe2_brg)

            melt_update = update_melt_reservoirs(n_fe3_melt, n_fe2_melt, fe3_part, fe2_part)
            n_fe3_melt, n_fe2_melt = melt_update["n_fe3_melt"], melt_update["n_fe2_melt"]

            ferric      = compute_ferric_fraction(melt_update)
            redox_ratio = compute_redox_ratio(ferric)
            ferric_frac     = ferric["ferric_frac"]
            redox_ratio_val = redox_ratio["redox_ratio"]
        else:
            # not in Eq 1-24: once the melt is gone there is nothing left to
            # partition, so freeze the last computed redox ratio and keep
            # evolving only fO2 through the cooling T(r), P(r) in Step 10
            if not melt_exhausted:
                print(f"  [melt exhausted at step {curr['step']}, t={curr['time_years']:.1f} yr "
                      f"-- freezing ferric_frac={ferric_frac:.6f} for remaining steps]")
            melt_exhausted = True
            fe3_part = fe2_part = melt_update = None

        # Step 10 (Hirschmann 2022 Eq 21): radial fO2 profile from the local
        # T(r) of every cell combined with the single, global, time-evolving
        # redox ratio (the ratio itself has no radial structure -- see
        # docstring; nor does this profile depend on P(r), since DeltaVdP=0)
        fO2_result     = compute_fO2(redox_ratio_val, curr, X)
        log10_fO2_cell = [lf / math.log(10) for lf in fO2_result["ln_fO2"]]

        # surface point of that same profile, evaluated at the true surface
        # (top _b node, P ~ 0) rather than cell 0 of the _s grid (~1-2 GPa),
        # then converted to Delta-IW via the Hirschmann (2021) buffer
        surf_snapshot  = {"temp": [curr["temp_surf"]], "pressure": [curr["pressure_surf"]]}
        fO2_surf       = compute_fO2(redox_ratio_val, surf_snapshot, X)
        log10_fO2_surf = fO2_surf["ln_fO2"][0] / math.log(10)
        dIW_surface    = log10_fO2_surf - iw_buffer_hirschmann2021(
            curr["temp_surf"], curr["pressure_surf"] / 1e9)

        # per-cell, per-timestep dump (Steps 3-10); melt_dist here is the
        # "(t-1)" cell state that fed this iteration's Step 5-6 partitioning
        n_FeT_melt_global = n_fe3_melt + n_fe2_melt
        for c in range(n_cells):
            csv_writer.writerow(_csv_format_row([
                curr["step"], curr["time_years"], c,
                curr["radius"][c], curr["pressure"][c], curr["temp"][c],
                prev["phi"][c], curr["phi"][c], solid["delta_phi"][c],
                solid["new_solid_frac"][c], solid["new_solid_mass"][c],
                melt_dist["M_melt_cell"][c],
                melt_dist["n_fe3_melt_cell"][c], melt_dist["n_fe2_melt_cell"][c],
                D_fe3_cell[c],
                fe3_part["delta_n_fe3_solid"][c] if fe3_part else float("nan"),
                fe2_part["delta_n_fe2_solid"][c] if fe2_part else float("nan"),
                n_fe3_melt, n_fe2_melt, n_FeT_melt_global,
                ferric_frac, redox_ratio_val,
                log10_fO2_cell[c],
                log10_fO2_surf, dIW_surface,
            ]))

        results["time_years"].append(curr["time_years"])
        results["step"].append(curr["step"])
        results["ferric_frac"].append(ferric_frac)
        results["redox_ratio"].append(redox_ratio_val)
        results["log10_fO2_cell"].append(log10_fO2_cell)
        results["log10_fO2_surf"].append(log10_fO2_surf)
        results["dIW_surface"].append(dIW_surface)

        if i <= verbose_steps and not melt_exhausted:
            _print_verbose_step(i, iron, fe_split, melt_dist, solid, fe3_part, fe2_part,
                                 melt_update, ferric_frac, redox_ratio_val, prev, curr,
                                 log10_fO2_cell, log10_fO2_surf, dIW_surface)
        else:
            print(f"{curr['step']:>7d}  {curr['time_years']:>14.1f}  {ferric_frac:>12.6f}  "
                  f"{redox_ratio_val:>12.6f}  {min(log10_fO2_cell):>13.3f}  "
                  f"{max(log10_fO2_cell):>13.3f}  {dIW_surface:>10.3f}")

        # Step 4 for the *next* iteration
        if not melt_exhausted:
            melt_dist = distribute_melt_composition(curr, n_fe3_melt, n_fe2_melt)

    csv_file.close()
    print(f"Wrote per-cell, per-timestep time series to {csv_path} "
          f"({len(snapshots) - 1} steps x {n_cells} cells)")

    return results, snapshots[0]


# ── main ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    files = sorted(
        glob.glob(os.path.join(DATA_DIR, "*.json")),
        key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
    )

    results, snap0 = run_redox_evolution(files, verbose_steps=VERBOSE_STEPS)

    print()
    print(f"Ran {len(results['step'])} crystallization steps "
          f"(t = {results['time_years'][0]:.1f} to {results['time_years'][-1]:.1f} yr)")

    # ── Plot 1: ferric fraction evolution in time ────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(results["time_years"], results["ferric_frac"], color="firebrick",
             label=r"ferric fraction  Fe$^{3+}$/Fe$_T$")
    ax1.axhline(f_0, color="gray", linestyle=":", linewidth=1, label=f"initial f$_0$ = {f_0}")
    ax1.set_xscale("log")
    ax1.set_xlabel("time (yr)")
    ax1.set_ylabel(r"Fe$^{3+}$/Fe$_T$ (melt, global)")
    ax1.set_title("Melt redox evolution during crystallization")
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)
    ax1.legend()
    plt.tight_layout()
    plt.savefig("ferric_fraction_vs_time.png", dpi=150)

    # ── Plot 1b: dIW at the true surface vs time ──────────────────────────────
    fig1b, ax1b = plt.subplots(figsize=(7, 4))
    ax1b.plot(results["time_years"], results["dIW_surface"], color="darkorange")
    ax1b.axhline(0, color="gray", linestyle=":", linewidth=1, label="IW")
    ax1b.set_xscale("log")
    ax1b.set_xlabel("time (yr)")
    ax1b.set_ylabel(r"$\Delta$IW at surface")
    ax1b.set_title("Surface redox state relative to IW\n"
                    "(fO2: Hirschmann 2022 Eq 21; IW buffer: Hirschmann 2021)")
    ax1b.grid(True, which="both", linestyle="--", alpha=0.4)
    ax1b.legend()
    plt.tight_layout()
    plt.savefig("dIW_surface_vs_time.png", dpi=150)

    # ── Plot 2: log10(fO2) radial profile vs pressure (depth) and time ────────
    n = len(snap0["phi"])
    pressure_GPa = [snap0["pressure"][c] / 1e9 for c in range(n)]
    Z = np.array(results["log10_fO2_cell"]).T   # shape (n_cells, n_timesteps)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    mesh = ax2.pcolormesh(results["time_years"], pressure_GPa, Z,
                           shading="auto", cmap="viridis")
    ax2.set_xscale("log")
    ax2.invert_yaxis()
    ax2.set_xlabel("time (yr)")
    ax2.set_ylabel("Pressure (GPa)")
    ax2.set_title(r"log$_{10}$(f$_{O_2}$) evolution with depth and time"
                  "\n(Hirschmann 2022 Eq 21, ΔVdP=0 — varies only through T(r))")
    cbar = fig2.colorbar(mesh, ax=ax2)
    cbar.set_label(r"log$_{10}$(f$_{O_2}$)")
    plt.tight_layout()
    plt.savefig("fO2_radial_profile.png", dpi=150)

    print("Saved ferric_fraction_vs_time.png, dIW_surface_vs_time.png "
          "and fO2_radial_profile.png")
