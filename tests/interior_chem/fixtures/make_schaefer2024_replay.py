"""Write the Schaefer et al. (2024) replay fixture used by test_redox.py.

Reads ``output.mat`` of the archived Earth code of Schaefer, Pahlevan and
Elkins-Tanton (2024, JGR Planets), Stanford Digital Repository (Schaefer
2023), which holds 1000 Monte Carlo fractional-crystallisation runs of a
1000 km magma ocean with initial Fe3+/FeT = 0.1 (``frac_xtl_start = 2``).
For one run it writes, per crystallisation step, the pressure, the
crystallised mass fraction of the magma ocean, the bulk solid/melt D of
FeO and FeO1.5 (mass concentration ratios over the layer mineralogy) and
the molar Fe3+/FeT of the residual liquid.

The default run, 570, is the run whose final Fe3+/FeT is the median of the
runs whose mass balance closes to 1e-9 at every step; in the other runs the
MATLAB solver left some steps unconverged.

Usage::

    python make_schaefer2024_replay.py path/to/output.mat [--run 570]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat

OUT = Path(__file__).with_name('schaefer2024_1000km_replay.csv')
# Oxide order in the archived arrays: FeO MgO SiO2 CaO Al2O3 FeO1.5
I_FE2, I_FE3 = 0, 5


def _bulk_solid(m: dict, i: int, j: int) -> np.ndarray:
    """Bulk oxide mass fractions of the solid of step ``i`` in run ``j``,
    weighted by the mineral fractions of the layer it formed in."""
    p_layers = m['P_layers'].ravel()
    layer = int(np.clip(np.searchsorted(p_layers, m['Psolid_pf'][i, j], 'left') - 1, 0, None))
    return m['minerals'][:, layer] @ m['Xel_xtl_pf'][:, :, i, j]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('output_mat', type=Path)
    parser.add_argument('--run', type=int, default=570)
    args = parser.parse_args()

    m = loadmat(args.output_mat)
    if int(m['frac_xtl_start'].item()) != 2 or float(m['Fe3_FeT'].item()) != 0.1:
        raise ValueError('expected the 1000 km run with initial Fe3+/FeT = 0.1')
    mw, liq, s_all = m['MW'].ravel(), m['Xmo_pf'], m['Xsol_pf']
    j = args.run

    rows = []
    for i in range(1, s_all.shape[0]):
        s = float(s_all[i, j])
        if s <= 0.0:
            break
        sol = _bulk_solid(m, i, j)
        prev, new = liq[i - 1, :, j], liq[i, :, j]
        for k in (I_FE2, I_FE3):
            if abs(prev[k] - (1 - s) * new[k] - s * sol[k]) > 1e-9 * prev[k]:
                raise ValueError(f'run {j} step {i}: mass balance does not close')
        n2, n3 = new[I_FE2] / mw[I_FE2], new[I_FE3] / mw[I_FE3]
        rows.append(
            (
                i,
                float(m['Psolid_pf'][i, j]),
                s,
                sol[I_FE2] / new[I_FE2],
                sol[I_FE3] / new[I_FE3],
                n3 / (n2 + n3),
            )
        )

    header = (
        f'Schaefer et al. (2024) archived output.mat, 1000 km magma ocean, run {j}\n'
        'step: crystallisation step; P_GPa: pressure of the new solid layer\n'
        's: solid mass formed / magma-ocean mass at the start of the step\n'
        'D_fe2, D_fe3: bulk solid/melt mass-concentration ratio of FeO, FeO1.5\n'
        'fe3_feT: molar Fe3+/FeT of the liquid after the step (0.1 before step 1)\n'
        'step,P_GPa,s,D_fe2,D_fe3,fe3_feT'
    )
    np.savetxt(OUT, rows, fmt=['%d'] + ['%.12e'] * 5, delimiter=',', header=header)
    print(f'wrote {len(rows)} steps to {OUT}')


if __name__ == '__main__':
    main()
