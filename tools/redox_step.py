"""
Step 1 — Load SPIDER JSON timesteps and compute new solid mass per cell.

Grid convention:
  _s  = staggered nodes (99 cells): pressure, mass, radius, phi_s, temp_s
  _b  = basic/boundary nodes (100 cells): phi_b, temperature, etc.

Cells are ordered top → bottom (index 0 = surface, index 98 = CMB).

Values in the JSON are stored as strings; multiply by the 'scaling'
field to get physical units (e.g. radius_s * scaling = metres).
"""
from __future__ import annotations

import glob
import json
import math
import os

import matplotlib.pyplot as plt

DATA_DIR = "output/data"

# ── fixed parameters ──────────────────────────────────────────────────────────
w_FeT  = 0.08       # bulk iron mass fraction in the melt (dimensionless)
mu_FeO = 0.07184    # molar mass of FeO in kg/mol (= 71.84 g/mol)
f_0        = 0.10   # initial ferric fraction Fe³⁺/FeT (Hirschmann 2022)
D_fe3_brg  = 0.75  # bridgmanite/melt partition coefficient for Fe³⁺
D_fe2_brg  = 0.85  # bridgmanite/melt partition coefficient for Fe²⁺

# BSE starting composition (wt%) — McDonough (2003) via Hirschmann (2022)
wt_FeO   = 7.82
wt_MgO   = 38.3
wt_SiO2  = 45.5
wt_CaO   = 3.58
wt_Al2O3 = 4.49
wt_FeO15 = 0.36   # FeO1.5
wt_Na2O  = 0.0    # not listed in BSE table
wt_K2O   = 0.0    # not listed in BSE table


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
        time_years  – simulation time in years
        step        – SPIDER step index
        phi         – melt fraction per cell (dimensionless)
        pressure    – pressure per cell (Pa)
        mass        – mass per cell (kg)
        radius      – radius per cell (m)
        temp        – temperature per cell (K)
    """
    with open(filepath) as f:
        raw = json.load(f)

    phi,      _          = _load_field(raw, "phi_s")
    pressure, P_scale    = _load_field(raw, "pressure_s")
    mass,     mass_scale = _load_field(raw, "mass_s")
    radius,   r_scale    = _load_field(raw, "radius_s")
    temp,     T_scale    = _load_field(raw, "temp_s")

    return {
        "time_years": raw["time_years"],
        "step":       raw["step"],
        "phi":        phi,
        "pressure":   [p * P_scale    for p in pressure],
        "mass":       [m * mass_scale for m in mass],
        "radius":     [r * r_scale    for r in radius],
        "temp":       [t * T_scale    for t in temp],
    }


def init_redox_state(n_cells):
    """
    Initialise per-cell redox arrays with placeholder values.

    fe3_melt  – mol fraction of Fe³⁺ in melt  (placeholder: 0.1)
    fe2_melt  – mol fraction of Fe²⁺ in melt  (placeholder: 0.9)
    fe3_solid – mol fraction of Fe³⁺ in solid (placeholder: 0.0, no solid yet)
    fe2_solid – mol fraction of Fe²⁺ in solid (placeholder: 0.0, no solid yet)
    """
    return {
        "fe3_melt":  [0.1] * n_cells,
        "fe2_melt":  [0.9] * n_cells,
        "fe3_solid": [0.0] * n_cells,
        "fe2_solid": [0.0] * n_cells,
    }


def split_iron_reservoirs(iron, f_0):
    """
    Split the total iron inventory into Fe³⁺ and Fe²⁺ (Eq. 2 and 3).

    Parameters
    ----------
    iron : dict   from init_iron_inventory()
    f_0  : float  initial ferric fraction Fe³⁺/FeT

    Returns
    -------
    dict with:
        n_fe3_melt_0  – initial Fe³⁺ moles in melt = f_0 * n_FeT_melt_0
        n_fe2_melt_0  – initial Fe²⁺ moles in melt = (1 - f_0) * n_FeT_melt_0
    """
    n_FeT = iron["n_FeT_melt_0"]
    return {
        "n_fe3_melt_0": f_0 * n_FeT,
        "n_fe2_melt_0": (1 - f_0) * n_FeT,
    }


def distribute_melt_composition(snapshot, n_fe3_melt, n_fe2_melt):
    """
    Redistribute global melt iron across cells in proportion to melt mass (Eq. 9-11).

    Parameters
    ----------
    snapshot   : dict   from read_spider_step(), provides phi and mass per cell
    n_fe3_melt : float  total Fe³⁺ moles in the global melt reservoir
    n_fe2_melt : float  total Fe²⁺ moles in the global melt reservoir

    Returns
    -------
    dict with per-cell arrays:
        M_melt_cell      – melt mass per cell (kg)
        M_melt_all       – total melt mass (kg)
        n_fe3_melt_cell  – Fe³⁺ moles per cell
        n_fe2_melt_cell  – Fe²⁺ moles per cell
    """
    n = len(snapshot["phi"])

    M_melt_cell = [snapshot["phi"][c] * snapshot["mass"][c]  for c in range(n)]  # Eq. 8
    M_melt_all  = sum(M_melt_cell)                                                # Eq. 9

    n_fe3_melt_cell = [(M_melt_cell[c] / M_melt_all) * n_fe3_melt  for c in range(n)]  # Eq. 10
    n_fe2_melt_cell = [(M_melt_cell[c] / M_melt_all) * n_fe2_melt  for c in range(n)]  # Eq. 11

    return {
        "M_melt_cell":     M_melt_cell,
        "M_melt_all":      M_melt_all,
        "n_fe3_melt_cell": n_fe3_melt_cell,
        "n_fe2_melt_cell": n_fe2_melt_cell,
    }


def compute_fe3_partitioning(melt_dist, solid, D_fe3_brg):
    """
    Compute Fe³⁺ moles entering the newly formed solid per cell (Eq. 12–14).

    Parameters
    ----------
    melt_dist  : dict   from distribute_melt_composition() — provides
                        n_fe3_melt_cell and M_melt_cell at t-1
    solid      : dict   from compute_new_solid() — provides new_solid_mass
    D_fe3_brg  : float  bridgmanite/melt partition coefficient for Fe³⁺

    Returns
    -------
    dict with per-cell array:
        delta_n_fe3_solid  – Fe³⁺ moles removed from melt into solid (Eq. 14)
    """
    n = len(melt_dist["n_fe3_melt_cell"])
    delta_n_fe3_solid = []

    for c in range(n):
        M_melt_i = melt_dist["M_melt_cell"][c]
        if M_melt_i > 0:
            C_fe3_melt_i      = melt_dist["n_fe3_melt_cell"][c] / M_melt_i          # Eq. 12
            delta_n_fe3_i     = D_fe3_brg * C_fe3_melt_i * solid["new_solid_mass"][c]  # Eq. 14
        else:
            delta_n_fe3_i = 0.0
        delta_n_fe3_solid.append(delta_n_fe3_i)

    return {"delta_n_fe3_solid": delta_n_fe3_solid}


def compute_fe2_partitioning(melt_dist, solid, D_fe2_brg):
    """
    Compute Fe²⁺ moles entering the newly formed solid per cell (Eq. 15–17).

    Parameters
    ----------
    melt_dist  : dict   from distribute_melt_composition() — provides
                        n_fe2_melt_cell and M_melt_cell at t-1
    solid      : dict   from compute_new_solid() — provides new_solid_mass
    D_fe2_brg  : float  bridgmanite/melt partition coefficient for Fe²⁺

    Returns
    -------
    dict with per-cell array:
        delta_n_fe2_solid  – Fe²⁺ moles removed from melt into solid (Eq. 17)
    """
    n = len(melt_dist["n_fe2_melt_cell"])
    delta_n_fe2_solid = []

    for c in range(n):
        M_melt_i = melt_dist["M_melt_cell"][c]
        if M_melt_i > 0:
            C_fe2_melt_i  = melt_dist["n_fe2_melt_cell"][c] / M_melt_i             # Eq. 15
            delta_n_fe2_i = D_fe2_brg * C_fe2_melt_i * solid["new_solid_mass"][c]  # Eq. 17
        else:
            delta_n_fe2_i = 0.0
        delta_n_fe2_solid.append(delta_n_fe2_i)

    return {"delta_n_fe2_solid": delta_n_fe2_solid}


def update_melt_reservoirs(n_fe3_melt_prev, n_fe2_melt_prev, fe3_part, fe2_part):
    """
    Update global melt Fe³⁺ and Fe²⁺ reservoirs after crystallization (Eq. 18–19).

    Parameters
    ----------
    n_fe3_melt_prev : float  Fe³⁺ moles in melt at t-1
    n_fe2_melt_prev : float  Fe²⁺ moles in melt at t-1
    fe3_part        : dict   from compute_fe3_partitioning()
    fe2_part        : dict   from compute_fe2_partitioning()

    Returns
    -------
    dict with:
        n_fe3_melt  – updated Fe³⁺ moles in melt at t (Eq. 18)
        n_fe2_melt  – updated Fe²⁺ moles in melt at t (Eq. 19)
    """
    n_fe3_melt = n_fe3_melt_prev - sum(fe3_part["delta_n_fe3_solid"])  # Eq. 18
    n_fe2_melt = n_fe2_melt_prev - sum(fe2_part["delta_n_fe2_solid"])  # Eq. 19

    return {
        "n_fe3_melt": n_fe3_melt,
        "n_fe2_melt": n_fe2_melt,
    }


def compute_ferric_fraction(melt_update):
    """
    Compute the updated total iron and ferric fraction of the melt (Eq. 20–21).

    Parameters
    ----------
    melt_update : dict   from update_melt_reservoirs()

    Returns
    -------
    dict with:
        n_FeT_melt    – total iron moles in melt (Eq. 20)
        ferric_frac   – Fe³⁺/FeT ratio of the melt (Eq. 21)
    """
    n_fe3 = melt_update["n_fe3_melt"]
    n_fe2 = melt_update["n_fe2_melt"]

    n_FeT_melt  = n_fe3 + n_fe2           # Eq. 20
    ferric_frac = n_fe3 / n_FeT_melt      # Eq. 21

    return {
        "n_FeT_melt":  n_FeT_melt,
        "ferric_frac": ferric_frac,
    }


def compute_redox_ratio(ferric):
    """
    Convert the ferric fraction to the Fe³⁺/Fe²⁺ mole fraction ratio (Eq. 24).

    Parameters
    ----------
    ferric : dict   from compute_ferric_fraction()

    Returns
    -------
    dict with:
        redox_ratio  – X_Fe3+ / X_Fe2+ = f / (1 - f)
    """
    f = ferric["ferric_frac"]
    return {"redox_ratio": f / (1.0 - f)}


def compute_mole_fractions(wt_FeO, wt_MgO, wt_SiO2, wt_CaO,
                           wt_Al2O3, wt_FeO15, wt_Na2O, wt_K2O):
    """
    Convert oxide wt% to mole fractions.

    Returns a dict with keys: FeO, MgO, SiO2, CaO, Al2O3, FeO15, Na2O, K2O.
    """
    M = {"FeO": 71.84, "MgO": 40.30, "SiO2": 60.08, "CaO": 56.08,
         "Al2O3": 101.96, "FeO15": 79.84, "Na2O": 61.98, "K2O": 94.20}
    wt = {"FeO": wt_FeO, "MgO": wt_MgO, "SiO2": wt_SiO2, "CaO": wt_CaO,
          "Al2O3": wt_Al2O3, "FeO15": wt_FeO15, "Na2O": wt_Na2O, "K2O": wt_K2O}
    n       = {k: wt[k] / M[k] for k in M}
    n_total = sum(n.values())
    return {k: v / n_total for k, v in n.items()}


def compute_fO2(redox_ratio_val, snapshot, X):
    """
    Compute ln(fO2) and fO2 per cell using Schaefer et al. (2024) (Eq. 25–26).

    Parameters
    ----------
    redox_ratio_val : float   X_Fe3+ / X_Fe2+ = f/(1-f) from Step 9
    snapshot        : dict    from read_spider_step(), provides T and P per cell
    X               : dict    oxide mole fractions from compute_mole_fractions()

    Notes
    -----
    Pressure is converted from Pa (SPIDER) to bar for the Schaefer formula.
    T0 = 1673 K is the reference temperature of Kress & Carmichael (1991).
    """
    T0      = 1673.0
    ln_r    = math.log(redox_ratio_val)
    X_FeO_T = X["FeO"] + X["FeO15"]    # total iron mole fraction

    ln_fO2_list = []
    fO2_list    = []

    for c in range(len(snapshot["temp"])):
        T = snapshot["temp"][c]
        P = snapshot["pressure"][c] / 1e5      # Pa → bar

        ln_fO2_c = (1.0 / 0.196) * (
            ln_r
            - 1.1492e4 / T
            + 6.675
            + 2.243  * X["Al2O3"]
            + 1.828  * X_FeO_T
            - 3.201  * X["CaO"]
            - 5.854  * X["Na2O"]
            - 6.215  * X["K2O"]
            + 3.36   * (1.0 - T0/T - math.log(T / T0))
            + 7.01e-7  * P / T
            + 1.54e-10 * (T - T0) * P / T
            - 3.85e-17 * P**2 / T
        )
        ln_fO2_list.append(ln_fO2_c)
        fO2_list.append(math.exp(ln_fO2_c))

    return {"ln_fO2": ln_fO2_list, "fO2": fO2_list}


def compute_new_solid(step_prev, step_curr):
    """
    Compute newly formed solid between two consecutive timesteps.

    Returns a dict with one entry per cell:
        delta_phi      – decrease in melt fraction (negative = remelting, clipped to 0)
        new_solid_frac – fraction of cell mass that solidified  (>= 0)
        new_solid_mass – mass of new solid in kg                (>= 0)
    """
    n = len(step_prev["phi"])

    delta_phi      = [step_prev["phi"][c] - step_curr["phi"][c]   for c in range(n)]
    new_solid_frac = [max(0.0, d)                                   for d in delta_phi]
    new_solid_mass = [step_curr["mass"][c] * new_solid_frac[c]     for c in range(n)]

    return {
        "delta_phi":      delta_phi,
        "new_solid_frac": new_solid_frac,
        "new_solid_mass": new_solid_mass,
    }


def init_iron_inventory(snapshot, w_FeT, mu_FeO):
    """
    Compute the initial total iron inventory in the melt (Eq. 1).

    Parameters
    ----------
    snapshot : dict   from read_spider_step(), used for M_melt_0
    w_FeT    : float  bulk iron mass fraction in the melt
    mu_FeO   : float  molar mass of FeO in kg/mol

    Returns
    -------
    dict with:
        M_melt_0       – total melt mass at t=0  (kg)
        m_FeT_melt_0   – total iron mass in melt (kg)   = w_FeT * M_melt_0
        n_FeT_melt_0   – total iron moles in melt (mol) = m_FeT_melt_0 / mu_FeO
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


# ── main: test with first two timesteps ──────────────────────────────────────

files = sorted(
    glob.glob(os.path.join(DATA_DIR, "*.json")),
    key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
)

prev = read_spider_step(os.path.join(DATA_DIR, "6098.json"))
curr = read_spider_step(os.path.join(DATA_DIR, "6398.json"))
solid = compute_new_solid(prev, curr)

n = len(prev["phi"])
redox = init_redox_state(n)

iron      = init_iron_inventory(prev, w_FeT, mu_FeO)
fe_split  = split_iron_reservoirs(iron, f_0)
melt_dist = distribute_melt_composition(prev,
                                        fe_split["n_fe3_melt_0"],
                                        fe_split["n_fe2_melt_0"])
fe3_part    = compute_fe3_partitioning(melt_dist, solid, D_fe3_brg)
fe2_part    = compute_fe2_partitioning(melt_dist, solid, D_fe2_brg)
melt_update  = update_melt_reservoirs(fe_split["n_fe3_melt_0"],
                                      fe_split["n_fe2_melt_0"],
                                      fe3_part, fe2_part)
ferric       = compute_ferric_fraction(melt_update)
redox_ratio  = compute_redox_ratio(ferric)
X            = compute_mole_fractions(wt_FeO, wt_MgO, wt_SiO2, wt_CaO,
                                      wt_Al2O3, wt_FeO15, wt_Na2O, wt_K2O)
fO2_result   = compute_fO2(redox_ratio["redox_ratio"], curr, X)

print(f"Snapshot 1: step={prev['step']},  time={prev['time_years']:.1f} yr")
print(f"Snapshot 2: step={curr['step']},  time={curr['time_years']:.1f} yr")
print(f"Cells: {n}")
print()
melt_mass  = [curr["phi"][c] * curr["mass"][c]           for c in range(n)]
solid_mass = [(1 - curr["phi"][c]) * curr["mass"][c]     for c in range(n)]
total_mass = [melt_mass[c] + solid_mass[c]               for c in range(n)]

print(f"{'cell':>5}  {'phi_prev':>9}  {'phi_curr':>9}  {'delta_phi':>10}  {'new_solid_frac':>15}  {'new_solid_mass (kg)':>20}  {'melt_mass (kg)':>16}  {'solid_mass (kg)':>16}  {'total_mass (kg)':>16}  {'fe3_melt':>10}  {'fe2_melt':>10}  {'fe3_solid':>10}  {'fe2_solid':>10}")
print("-" * 180)
for c in range(n):
    print(f"{c:>5}  {prev['phi'][c]:>9.4f}  {curr['phi'][c]:>9.4f}  "
          f"{solid['delta_phi'][c]:>10.4f}  {solid['new_solid_frac'][c]:>15.4f}  "
          f"{solid['new_solid_mass'][c]:>20.3e}  {melt_mass[c]:>16.3e}  "
          f"{solid_mass[c]:>16.3e}  {total_mass[c]:>16.3e}  "
          f"{melt_dist['n_fe3_melt_cell'][c]:>10.3e}  {melt_dist['n_fe2_melt_cell'][c]:>10.3e}  "
          f"{redox['fe3_solid'][c]:>10.2f}  {redox['fe2_solid'][c]:>10.2f}")

print()
print(f"Cells with new solid : {sum(1 for m in solid['new_solid_mass'] if m > 0)}")
print(f"Total new solid mass : {sum(solid['new_solid_mass']):.3e} kg")

print()
print("── Step 1: initial iron inventory ───────────────────────")
print(f"  M_melt_0     = {iron['M_melt_0']:.4e} kg")
print(f"  m_FeT_melt_0 = {iron['m_FeT_melt_0']:.4e} kg")
print(f"  n_FeT_melt_0 = {iron['n_FeT_melt_0']:.4e} mol")

print()
print("── Step 2: initial Fe³⁺ / Fe²⁺ reservoirs ──────────────")
print(f"  f_0           = {f_0}")
print(f"  n_fe3_melt_0  = {fe_split['n_fe3_melt_0']:.4e} mol")
print(f"  n_fe2_melt_0  = {fe_split['n_fe2_melt_0']:.4e} mol")
print(f"  check (sum)   = {fe_split['n_fe3_melt_0'] + fe_split['n_fe2_melt_0']:.4e} mol  (should equal n_FeT_melt_0)")

print()
print("── Step 4: melt composition distributed across cells ────")
print(f"  M_melt_all              = {melt_dist['M_melt_all']:.4e} kg")
print(f"  sum n_fe3_melt_cell     = {sum(melt_dist['n_fe3_melt_cell']):.4e} mol  (should equal n_fe3_melt_0)")
print(f"  sum n_fe2_melt_cell     = {sum(melt_dist['n_fe2_melt_cell']):.4e} mol  (should equal n_fe2_melt_0)")

print()
print("── Step 5: Fe³⁺ partitioning into solid ─────────────────")
print(f"  D_fe3_brg                  = {D_fe3_brg}")
print(f"  total delta_n_fe3_solid    = {sum(fe3_part['delta_n_fe3_solid']):.4e} mol")
print(f"  cells with Fe³⁺ → solid   = {sum(1 for x in fe3_part['delta_n_fe3_solid'] if x > 0)}")

print()
print("── Step 6: Fe²⁺ partitioning into solid ─────────────────")
print(f"  D_fe2_brg                  = {D_fe2_brg}")
print(f"  total delta_n_fe2_solid    = {sum(fe2_part['delta_n_fe2_solid']):.4e} mol")
print(f"  cells with Fe²⁺ → solid   = {sum(1 for x in fe2_part['delta_n_fe2_solid'] if x > 0)}")

print()
print("── Step 7: updated global melt reservoirs ───────────────")
print(f"  n_fe3_melt (prev) = {fe_split['n_fe3_melt_0']:.4e} mol")
print(f"  n_fe3_melt (curr) = {melt_update['n_fe3_melt']:.4e} mol  (Δ = -{sum(fe3_part['delta_n_fe3_solid']):.4e})")
print(f"  n_fe2_melt (prev) = {fe_split['n_fe2_melt_0']:.4e} mol")
print(f"  n_fe2_melt (curr) = {melt_update['n_fe2_melt']:.4e} mol  (Δ = -{sum(fe2_part['delta_n_fe2_solid']):.4e})")

print()
print("── Step 8: updated ferric fraction ──────────────────────")
print(f"  n_FeT_melt  = {ferric['n_FeT_melt']:.4e} mol")
print(f"  ferric_frac = {ferric['ferric_frac']:.6f}  (initial f_0 = {f_0})")

print()
print("── Step 9: Fe³⁺/Fe²⁺ redox ratio ───────────────────────")
print(f"  f           = {ferric['ferric_frac']:.6f}")
print(f"  f / (1-f)   = {redox_ratio['redox_ratio']:.6f}")

print()
print("── Step 10: oxygen fugacity (Schaefer et al. 2024) ──────")
log10_fO2 = [lf / math.log(10) for lf in fO2_result["ln_fO2"]]
print(f"  Mole fractions (BSE): X_Al2O3={X['Al2O3']:.4f}  X_FeO_T={X['FeO']+X['FeO15']:.4f}  X_CaO={X['CaO']:.4f}")
print(f"  log10(fO2) — top cell (c=0) : {log10_fO2[0]:.3f}")
print(f"  log10(fO2) — mid cell (c=49): {log10_fO2[49]:.3f}")
print(f"  log10(fO2) — bot cell (c=98): {log10_fO2[98]:.3f}")
print(f"  log10(fO2) — min : {min(log10_fO2):.3f}")
print(f"  log10(fO2) — max : {max(log10_fO2):.3f}")

# ── plot: log10(fO2) vs pressure ─────────────────────────────────────────────
pressure_GPa = [curr["pressure"][c] / 1e9 for c in range(n)]

fig, ax = plt.subplots(figsize=(5, 7))
ax.plot(fO2_result["fO2"], pressure_GPa, color="steelblue", linewidth=1.5)
ax.set_xlabel("f$_{O_2}$")
ax.set_ylabel("Pressure (GPa)")
ax.set_title(f"fO₂ profile  |  t = {curr['time_years']:.0f} yr")
ax.invert_yaxis()   # surface (low P) at top, CMB (high P) at bottom
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("fO2_vs_depth.png", dpi=150)
plt.show()
print("Plot saved to fO2_vs_depth.png")
