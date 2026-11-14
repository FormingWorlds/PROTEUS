"""Store and measure the core-mantle boundary (CMB) heat flux of the mixing-length validation runs.

Each case is one PROTEUS run in the folder of its name under the output directory. The 10
cases of 1 Earth mass run ``input/tutorials/tutorial_lid_seal.toml`` to 100 kyr
(``params.stop.time.maximum = 1e5``) with the keys of their description set:
``interior_energetics.num_levels`` (80 or 160), ``interior_energetics.aragog.cmb_cell_thickness``
(0 for the uniform mesh, 1500 or 3000 m), ``interior_energetics.mixing_length``, and for the
rheology pair ``interior_energetics.aragog.rheology.enabled = true``; the round-off case also
sets ``planet.mass_tot = 1.000000000001``. The 2 cases of 1.5 Earth masses run
``planet_1p5_earth.toml`` of this folder to 300 kyr under the default mixing length; the
default-mesh case leaves out its ``cmb_cell_thickness = 0``. ``main`` writes ``Time`` [yr],
``T_cmb`` [K] and ``F_cmb`` [W m-2, positive out of the core] of every helpfile row of each run
into one compressed ``.npz``; ``metrics`` measures a case from it.

Usage::

    python tools/verification/mixing_length/cmb_flux.py OUTPUT_DIR NPZ --commit SHA --aragog SHA
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from proteus.utils.coupler import read_helpfile_table

# case: description; the run folder has the name of the case
CASES = {
    'u80_nearest': 'uniform mesh, 80 nodes, nearest',
    'u80_constant': 'uniform mesh, 80 nodes, constant',
    'u160_nearest': 'uniform mesh, 160 nodes, nearest',
    'u160_constant': 'uniform mesh, 160 nodes, constant',
    'c3000_nearest': '3000 m CMB cell, 80 nodes, nearest',
    'c3000_constant': '3000 m CMB cell, 80 nodes, constant',
    'c1500_constant': '1500 m CMB cell, 80 nodes, constant',
    'c3000_constant_roundoff': '3000 m CMB cell, 80 nodes, constant, planet mass changed by 1e-12',
    'rheology_nearest': 'rheology on, 3000 m CMB cell, 80 nodes, nearest',
    'rheology_constant': 'rheology on, 3000 m CMB cell, 80 nodes, constant',
    'm1p5_uniform': '1.5 Earth masses, uniform mesh, default',
    'm1p5_default': '1.5 Earth masses, default mesh, default',
}
COLUMNS = ('Time', 'T_cmb', 'F_cmb')
LOW_FLUX = 100.0  # W m-2; a row below it counts as a dip of the CMB flux
T_COUNT = 1.0e4  # yr; rows after this time are counted (the start transient is earlier)
T_STEP = 2.0e3  # yr; steps of T_cmb after this time are measured


def read_case(path, case: str) -> pd.DataFrame:
    """Read the rows of one case from the stored ``.npz``."""
    with np.load(path) as data:
        return pd.DataFrame({c: data[f'{case}.{c}'] for c in COLUMNS})


def read_source(path) -> str:
    """Return the commits the stored runs were made with."""
    with np.load(path) as data:
        return str(data['source'])


def metrics(frame: pd.DataFrame) -> dict:
    """Measure one case: the rows of F_cmb below LOW_FLUX after T_COUNT, the longest run of
    consecutive such rows with its span in yr, the rows below 0 and with |F_cmb| < 1 W m-2, the lowest F_cmb,
    the largest drop and rise of T_cmb between rows after T_STEP, and the end state."""
    t, f, temp = (frame[c].to_numpy(float) for c in ('Time', 'F_cmb', 'T_cmb'))
    late = t > T_COUNT
    low = late & (f < LOW_FLUX)
    longest = run = 0
    span = 0.0  # yr, from the row before the longest run to its last row
    for i, flag in enumerate(low):
        run = run + 1 if flag else 0
        if run > longest:
            longest, span = run, float(t[i] - t[max(i - run, 0)])
    step = np.diff(temp)[t[1:] > T_STEP]
    k = int(np.argmin(np.where(late, f, np.inf)))
    return {
        'n_low': int(low.sum()),
        'longest_low': longest,
        'longest_low_span': span,
        'dips': [(float(a), float(b)) for a, b in zip(t[low], f[low])],
        'n_negative': int((late & (f < 0)).sum()),
        'n_below_1': int((late & (np.abs(f) < 1.0)).sum()),
        'F_min': float(f[k]),
        'F_min_time': float(t[k]),
        'worst_drop': float(step.min()),
        'worst_rise': float(step.max()),
        'T_end': float(temp[-1]),
        't_end': float(t[-1]),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('output', type=Path, help='folder that holds the run folders')
    ap.add_argument('npz', type=Path)
    ap.add_argument('--commit', required=True, help='PROTEUS commit of the runs')
    ap.add_argument('--aragog', required=True, help='aragog commit of the runs')
    a = ap.parse_args(argv)
    arrays = {'source': np.array(f'PROTEUS commit {a.commit}, aragog commit {a.aragog}')}
    for case in CASES:
        hf = read_helpfile_table(a.output / case / 'runtime_helpfile.csv')
        arrays |= {f'{case}.{c}': hf[c].to_numpy(float) for c in COLUMNS}
    a.npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.npz, **arrays)
    print(
        'cases', len(CASES), 'values', sum(v.size for k, v in arrays.items() if k != 'source')
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
