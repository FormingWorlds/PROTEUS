"""Quick verification that the new liquidus_super IC mode resolves the
expected CMB anchor, without running a full PROTEUS simulation.

Loads the smoke config, builds a synthetic hf_row with a realistic 1 M_E
P_cmb, and prints:
- The Fei+2021 liquidus temperature at that P_cmb
- The full anchor T_cmb = T_liq + delta_T_super
- The Zalmoxis-side mapped temperature_mode
- The energetics-side computed initial entropy (when EOS files available)

Exit 0 on success; non-zero if an unexpected exception bubbles out.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / 'input' / 'chili' / 'smoke_liquidus_super_paleos_1me.toml'


def main() -> int:
    from zalmoxis.melting_curves import paleos_liquidus

    from proteus.config import read_config_object
    from proteus.interior_struct.zalmoxis import (
        _resolve_zalmoxis_cmb_temperature,
        _resolve_zalmoxis_temperature_mode,
    )

    print(f'Loading config: {CONFIG}')
    cfg = read_config_object(CONFIG)
    print(f'  temperature_mode  = {cfg.planet.temperature_mode!r}')
    print(f'  delta_T_super     = {cfg.planet.delta_T_super} K')
    print(f'  tcmb_init (unused) = {cfg.planet.tcmb_init} K')
    print(f'  tsurf_init (unused) = {cfg.planet.tsurf_init} K')

    # 1 M_E reference P_cmb
    P_cmb_ref = 135e9
    hf_row = {'P_cmb': P_cmb_ref}

    print('\n--- temperature_mode mapping ---')
    mapped = _resolve_zalmoxis_temperature_mode(cfg.planet.temperature_mode)
    print(f'  PROTEUS mode {cfg.planet.temperature_mode!r} '
          f'-> Zalmoxis mode {mapped!r}')
    assert mapped == 'adiabatic_from_cmb', f'unexpected mapping: {mapped}'

    print('\n--- CMB anchor temperature at P_cmb = 135 GPa ---')
    T_liq = float(paleos_liquidus(P_cmb_ref))
    print(f'  T_liq_Fei2021(135 GPa)   = {T_liq:.1f} K')
    T_anchor = _resolve_zalmoxis_cmb_temperature(
        cfg, hf_row, cfg.planet.temperature_mode,
    )
    print(f'  T_cmb anchor (liquidus+delta) = {T_anchor:.1f} K')
    assert abs(T_anchor - (T_liq + cfg.planet.delta_T_super)) < 1e-6

    print('\n--- First-call fallback (P_cmb missing) ---')
    T_anchor_fallback = _resolve_zalmoxis_cmb_temperature(
        cfg, {}, cfg.planet.temperature_mode,
    )
    print(f'  T_cmb anchor (135 GPa fallback) = {T_anchor_fallback:.1f} K')
    assert abs(T_anchor_fallback - T_anchor) < 1e-6, (
        'fallback should match explicit 135 GPa case'
    )

    print('\n--- compute_initial_entropy (PALEOS-2phase path) ---')
    try:
        from proteus.interior_energetics.common import compute_initial_entropy
        S = compute_initial_entropy(cfg, hf_row=hf_row, fallback=3200.0)
        print(f'  S_target = {S:.1f} J/(kg.K) at P=135 GPa, '
              f'T_anchor={T_anchor:.0f} K')
        # Sanity: a fully-molten state at ~6440 K, 135 GPa should give
        # S in the 3000-4500 J/(kg.K) range for PALEOS WB17 tables.
        # Looser bound to allow for either anchoring convention.
        if not (2500.0 < S < 5500.0):
            print(f'  WARNING: S={S:.1f} outside expected range (2500, 5500)')
    except Exception as e:
        print(f'  compute_initial_entropy raised: {type(e).__name__}: {e}')
        print('  (this is acceptable when EOS data is not pre-installed; '
              'the runtime path will run during the actual smoke launch)')

    print('\nAll structural checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
