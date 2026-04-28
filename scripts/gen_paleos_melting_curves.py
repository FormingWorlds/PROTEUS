"""Generate PALEOS-Fei2021 melting-curve tables for FWL_DATA.

Produces a melting-curve directory in
``$FWL_DATA/interior_lookup_tables/Melting_curves/PALEOS-Fei2021/``
that mirrors the layout of the existing ``Monteux-600`` and
``Wolf_Bower+2018`` directories. Both Aragog (P-T tables for the
mushy-zone lever rule) and SPIDER (P-S tables) read these files.

Why this exists:
  PROTEUS+Aragog currently auto-generates PALEOS-derived solidus and
  liquidus on the fly when the mantle EOS is one of PALEOS:*,
  PALEOS-2phase:*, etc. (see proteus.interior_energetics.aragog around
  line 540). For the WB17-vs-PALEOS paper comparison we want WB17 runs
  (SPIDER static structure + WolfBower2018_MgSiO3 EOS) to use the
  *same* melting curves as PALEOS, so the only physical difference
  between the two pipelines is the EoS bookkeeping (S_0 anchoring),
  not the phase boundary.

  Pre-generating a shared FWL_DATA directory is the EoS-agnostic
  approach: WB17 configs simply set ``melting_dir = "PALEOS-Fei2021"``
  and the existing PROTEUS+Aragog pre-generated-table path is reused.

Liquidus source: ``zalmoxis.melting_curves.paleos_liquidus``,
                 piecewise Belonoshko+2005 / Fei+2021 (the same
                 function the PALEOS auto-generation uses internally,
                 so PALEOS and WB17 runs see byte-identical T_liq(P).
Solidus: liquidus * mushy_zone_factor. Default 0.80 matches the
         CHILI PALEOS configs.

Pressure grid: np.logspace(8, 12, 500), i.e. 0.1 to 1000 GPa with 500
points. Identical to the auto-gen path so the on-the-fly PALEOS
files and these pre-generated files agree to the last bit.

Output files (per format used by Aragog/SPIDER readers):
    liquidus_P-T.dat   -- two columns: P [Pa], T [K], 500 rows
    solidus_P-T.dat    -- two columns: P [Pa], T [K], 500 rows

P-S variants (liquidus_P-S.dat, solidus_P-S.dat) are NOT generated
here: those depend on the entropy EOS and are typically produced
separately when SPIDER P-S inversion is needed. For the Aragog-only
path used by the paper-comparison runs, P-T tables are sufficient.

Usage:
    python scripts/gen_paleos_melting_curves.py
        [--out DIR] [--mushy 0.80] [--n-points 500]
        [--p-min 1e8] [--p-max 1e12]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    fwl = os.environ.get('FWL_DATA', '/Users/timlichtenberg/work/fwl_data')
    default_out = Path(fwl) / 'interior_lookup_tables' / 'Melting_curves' / \
                   'PALEOS-Fei2021'
    p.add_argument('--out', type=Path, default=default_out,
                    help='Output melting-curve directory')
    p.add_argument('--mushy', type=float, default=0.80,
                    help='mushy_zone_factor: T_solidus = T_liquidus * mushy')
    p.add_argument('--n-points', type=int, default=500,
                    help='Number of pressure samples')
    p.add_argument('--p-min', type=float, default=1.0e8,
                    help='Minimum pressure [Pa]')
    p.add_argument('--p-max', type=float, default=1.0e12,
                    help='Maximum pressure [Pa]')
    args = p.parse_args()

    if not (0 < args.mushy <= 1):
        print(f'mushy must be in (0, 1]; got {args.mushy}', file=sys.stderr)
        return 2

    try:
        from zalmoxis.melting_curves import paleos_liquidus
    except ImportError as e:
        print(f'Cannot import zalmoxis.melting_curves: {e}', file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    P = np.logspace(np.log10(args.p_min), np.log10(args.p_max), args.n_points)
    T_liq = np.array([float(paleos_liquidus(p_i)) for p_i in P])
    T_sol = T_liq * args.mushy

    if not np.all(np.isfinite(T_liq)) or not np.all(T_liq > 0):
        print('paleos_liquidus produced non-finite or non-positive values',
              file=sys.stderr)
        return 1
    if not np.all(np.diff(T_liq) > 0):
        print('paleos_liquidus is not monotonically increasing with P',
              file=sys.stderr)
        return 1

    liq_path = args.out / 'liquidus_P-T.dat'
    sol_path = args.out / 'solidus_P-T.dat'
    np.savetxt(str(liq_path), np.column_stack([P, T_liq]),
               header='pressure temperature  (PALEOS Fei+2021 liquidus)',
               comments='#')
    np.savetxt(str(sol_path), np.column_stack([P, T_sol]),
               header=f'pressure temperature  (PALEOS Fei+2021 liquidus '
                       f'* mushy_zone_factor={args.mushy})',
               comments='#')

    # Sanity print at three reference pressures
    print(f'wrote {liq_path}')
    print(f'wrote {sol_path}')
    print()
    print(f'  mushy_zone_factor = {args.mushy}')
    print(f'  pressure grid: {args.p_min:.0e} -> {args.p_max:.0e} Pa, '
          f'{args.n_points} points')
    print()
    for p_ref in (1e9, 1e10, 1.35e11, 4e11):
        T_l = float(paleos_liquidus(p_ref))
        T_s = T_l * args.mushy
        print(f'  P = {p_ref:.2e} Pa:  T_sol = {T_s:7.1f} K   '
              f'T_liq = {T_l:7.1f} K   width = {T_l - T_s:.1f} K')

    return 0


if __name__ == '__main__':
    sys.exit(main())
