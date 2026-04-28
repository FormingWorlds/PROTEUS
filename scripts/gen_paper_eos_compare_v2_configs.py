"""Generate 4 paper-comparison configs (v2) using the EoS-agnostic
liquidus_super IC mode + matched PALEOS-Fei2021 melting curves.

Produces:
- 2 EoS variants:
  - PALEOS-2phase:MgSiO3 -> Zalmoxis structure (1 M_E)
  - WolfBower2018:MgSiO3 -> SPIDER static structure (canonical CHILI
    WB17 pairing; Zalmoxis basin-attractor failure at 1 M_E
    documented in closed_2026_04_26_aragog_zalmoxis_*)
- x 2 super-liquidus offsets:
  - hot:  delta_T_super = 500 K  (standard fully-molten IC)
  - warm: delta_T_super = 100 K  (near-liquidus IC)
= 4 configs in input/chili/paper_eos_compare_v2_*.toml

What's different from v1 (paper_eos_compare_*_{isen,adiab}.toml):

(1) IC mode is now `liquidus_super`. The CMB anchor is
    T_cmb = T_liq_Fei2021(P_cmb) + delta_T_super, where T_liq_Fei2021
    is the third-party Belonoshko+2005 / Fei+2021 piecewise melting
    curve shared with PALEOS internally. Because Fei+2021 is an
    EoS-independent calorimetric reference, the IC anchor does NOT
    bake in either the WB17 (S_0=0) or the PALEOS (Stebbins-anchored)
    entropy convention. v1's `isentropic` and `adiabatic` (== "adiabatic
    surface-anchored") modes both shifted the IC entropy across EoS
    by ~5000 J/(kg.K) due to the bookkeeping mismatch.

(2) WB17 runs use `melting_dir = "PALEOS-Fei2021"` instead of
    `"Monteux-600"`, so Aragog's lever-rule mushy zone uses the same
    solidus/liquidus T(P) as the PALEOS auto-generated curves. v1
    had Monteux-600 vs PALEOS-derived T(P), which contaminated
    melt-fraction comparisons across EoS.

The remaining residual: structure-side asymmetry between Zalmoxis
(self-consistent EOS) and SPIDER static (Adams-Williamson) is < 1%
on R_int/M_int and was previously confirmed to not contaminate the
6x cooling-timescale signal. After (1) and (2), the surviving
difference between PALEOS and WB17 paper runs at matched IC and
matched phase boundaries is purely the Aragog cooling-integrator
EoS bookkeeping.

Usage: python scripts/gen_paper_eos_compare_v2_configs.py

Prerequisite: run `python scripts/gen_paleos_melting_curves.py` once
on every machine that hosts FWL_DATA, to populate
$FWL_DATA/interior_lookup_tables/Melting_curves/PALEOS-Fei2021/.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PALEOS_BASE = REPO / "input" / "chili" / "chili_paleos_v1_1_0_1me_150res.toml"
OUT_DIR = REPO / "input" / "chili"

# Two IC variants for the v2 paper rerun. Both use liquidus_super; the
# only difference is delta_T_super, the super-liquidus offset at the
# CMB. 500 K = comfortably molten across the magma-ocean range
# (matches the v1 hot anchor heuristic); 100 K = near-liquidus, where
# the lever-rule mushy zone is most sensitive to the EoS bookkeeping.
IC_HOT = {
    "label": "hot",
    "delta_T_super": 500.0,
}
IC_WARM = {
    "label": "warm",
    "delta_T_super": 100.0,
}


# The Zalmoxis interior_struct block in the PALEOS base. Identical to
# the v1 generator. PALEOS configs keep this as-is; the structure-side
# T(r) integrates with the Fei-derived CMB anchor handled by
# proteus.interior_struct.zalmoxis._resolve_zalmoxis_cmb_temperature.
PALEOS_STRUCT_BLOCK = """[interior_struct]
core_frac = 0.325
core_frac_mode = "mass"
module = "zalmoxis"
core_density = "self"
core_heatcap = "self"
melting_dir = "none"
eos_dir = "none"

[interior_struct.zalmoxis]
core_eos = "PALEOS:iron"
mantle_eos = "PALEOS-2phase:MgSiO3"
ice_layer_eos = "none"
mushy_zone_factor = 0.8
mantle_mass_fraction = 0.0
num_levels = 150
solver_tol_outer = 0.003
solver_tol_inner = 0.0001
solver_max_iter_outer = 100
solver_max_iter_inner = 100
update_interval = 50000.0
update_min_interval = 100.0
update_dtmagma_frac = 0.05
update_dphi_abs = 0.05
update_stale_ceiling = 25000.0
mesh_max_shift = 0.05
mesh_convergence_interval = 10.0
equilibrate_init = true
equilibrate_max_iter = 15
equilibrate_tol = 0.01
dry_mantle = true
lookup_nP = 1350
lookup_nS = 280
global_miscibility = false
miscibility_max_iter = 10
miscibility_tol = 0.01
use_jax = true
use_anderson = false
outer_solver = "newton"
newton_max_iter = 30
newton_tol = 0.0001
newton_relative_tolerance = 1e-09
newton_absolute_tolerance = 1e-10
"""

# SPIDER static block for WB17. Differs from v1 in melting_dir only:
# now points at the pre-generated PALEOS-Fei2021 directory. eos_dir
# stays at WolfBower2018_MgSiO3 (we are explicitly comparing the WB17
# liquid EOS bookkeeping against PALEOS while neutralising the phase
# boundary).
SPIDER_STRUCT_BLOCK = """[interior_struct]
core_frac = 0.55
core_frac_mode = "radius"
module = "spider"
core_density = 10738.33
core_heatcap = 880.0
melting_dir = "PALEOS-Fei2021"
eos_dir = "WolfBower2018_MgSiO3"
"""


def patch_common(text: str, name: str, ic: dict) -> str:
    """Apply IC + output-path patches that are identical for both EoS variants.

    The base config (chili_paleos_v1_1_0_1me_150res.toml) has the v1-style
    [planet] block with temperature_mode = "adiabatic" and explicit
    tsurf_init / tcmb_init / ini_entropy. We rewrite the IC scalars to
    use liquidus_super + delta_T_super; the legacy fields are left in
    place but ignored at runtime when temperature_mode = liquidus_super.
    """
    repls = [
        # output path
        ('path = "chili_paleos_v1_1_0_1me_150res"', f'path = "{name}"'),
        # temperature_mode -> liquidus_super
        ('temperature_mode = "adiabatic"', 'temperature_mode = "liquidus_super"'),
    ]
    for old, new in repls:
        if old not in text:
            raise RuntimeError(f"expected substring not found in base: {old!r}")
        text = text.replace(old, new, 1)

    # Insert delta_T_super line. Place it directly after temperature_mode.
    inserted = (
        'temperature_mode = "liquidus_super"\n'
        f'delta_T_super = {ic["delta_T_super"]}'
    )
    if 'temperature_mode = "liquidus_super"' not in text:
        raise RuntimeError('temperature_mode patch did not land')
    if 'delta_T_super' in text:
        raise RuntimeError('delta_T_super already present in base; refusing to patch')
    text = text.replace(
        'temperature_mode = "liquidus_super"',
        inserted,
        1,
    )
    return text


def make_paleos_config(name: str, ic: dict, base_text: str) -> str:
    """PALEOS variant: keep Zalmoxis structure block as-is."""
    return patch_common(base_text, name, ic)


def make_wb17_config(name: str, ic: dict, base_text: str) -> str:
    """WB17 variant: swap Zalmoxis structure block for SPIDER static block."""
    text = patch_common(base_text, name, ic)
    if PALEOS_STRUCT_BLOCK not in text:
        raise RuntimeError(
            "PALEOS Zalmoxis structure block not found verbatim in base config; "
            "the generator's snapshot is out of sync with input/chili/chili_paleos_v1_1_0_1me_150res.toml"
        )
    text = text.replace(PALEOS_STRUCT_BLOCK, SPIDER_STRUCT_BLOCK, 1)
    return text


def main() -> None:
    if not PALEOS_BASE.is_file():
        raise SystemExit(f"PALEOS base config not found: {PALEOS_BASE}")
    base_text = PALEOS_BASE.read_text()

    written = []
    for ic in (IC_HOT, IC_WARM):
        for eos_key in ("wb17", "paleos"):
            name = f"paper_eos_compare_v2_{eos_key}_1me_{ic['label']}_lqdsupr"
            out = OUT_DIR / f"{name}.toml"
            if eos_key == "paleos":
                out.write_text(make_paleos_config(name, ic, base_text))
            else:
                out.write_text(make_wb17_config(name, ic, base_text))
            written.append(out.name)
            print(f"wrote {out.relative_to(REPO)}")

    print(f"\ndone: {len(written)} configs in {OUT_DIR.relative_to(REPO)}")
    print()
    print("Reminder: run scripts/gen_paleos_melting_curves.py first if you")
    print("haven't already populated the PALEOS-Fei2021 directory in FWL_DATA.")


if __name__ == "__main__":
    main()
