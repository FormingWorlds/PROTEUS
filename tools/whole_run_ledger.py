"""Print the whole-run energy ledger of a PROTEUS run.

Usage: ``python tools/whole_run_ledger.py RUN [--start ROW] [--end ROW] [--core-bc NAME]``

RUN is an output folder with ``runtime_helpfile.csv``; ``core_bc`` and ``core_tfac_avg`` come from its
``init_coupler.toml`` unless ``--core-bc`` is given. The terms and residuals are those of
``proteus.utils.energy_ledger.whole_run_ledger``, summed over the window and given relative to the heat
to the atmosphere L_int.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

from proteus.utils.coupler import read_helpfile_table
from proteus.utils.energy_ledger import Ledger, whole_run_ledger


def run_settings(run: Path, core_bc: str | None) -> tuple[str, float]:
    """The core boundary condition and core temperature factor of a run folder."""
    cfg = run / 'init_coupler.toml'
    ie = tomllib.loads(cfg.read_text()).get('interior_energetics', {}) if cfg.exists() else {}
    bc = core_bc or ie.get('aragog', {}).get('core_bc')
    if bc is None:
        raise ValueError(f'{cfg} names no aragog core_bc; pass --core-bc')
    return bc, float(ie.get('core_tfac_avg', 1.147))


def report(led: Ledger) -> list[str]:
    """Lines of the report: the summed terms and residuals, the core residual by step class, the
    step counts by class and the mantle mass change."""
    L = float(led.terms['L_int'].sum())
    rel = (lambda x: x / L) if L else (lambda x: float('nan'))
    lines = [f'steps {led.R_p.size}, t {led.time[0]:.6g} to {led.time[-1]:.6g} yr']
    for k in ('S_state', 'L_int', 'C_cmb', 'Q', 'I_mantle', 'D_core', 'B_core', 'E_atm'):
        lines.append(f'{k:10s} {led.terms[k].sum():+.6e} J')
    for k in ('R_m', 'R_c', 'G', 'R_p'):
        x = float(getattr(led, k).sum())
        lines.append(f'{k:10s} {x:+.6e} J  ({rel(x):+.3e} of L_int)')
    split = led.split('R_c')
    lines.append('R_c by step class: ' + ', '.join(f'{c} {v:+.4e} J' for c, v in split.items()))
    counts = ', '.join(f'{c} {int(v.sum())}' for c, v in led.classes.items())
    lines.append(f'steps by class: {counts}')
    lines.append(
        f'closure R_p/T {led.closure:+.3e} (T {led.throughput:.4e} J); compression work '
        f'(not summed) {led.terms["compression"].sum():+.4e} J'
    )
    lines.append(
        f'mantle mass change {led.dM_mantle.sum():+.4e} kg in {int((led.dM_mantle != 0).sum())} steps '
        '(energy not booked: it depends on the zero of the EOS energy)'
    )
    return lines


def main(argv: list[str] | None = None) -> int:
    """Command entry point; returns the exit code."""
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('run', type=Path)
    p.add_argument('--start', type=int, default=2)
    p.add_argument('--end', type=int, default=None)
    p.add_argument('--core-bc', default=None)
    a = p.parse_args(argv)
    try:
        core_bc, tfac = run_settings(a.run, a.core_bc)
        hf = read_helpfile_table(a.run / 'runtime_helpfile.csv')
        led = whole_run_ledger(hf, core_bc, tfac, start_row=a.start, end_row=a.end)
    except (OSError, ValueError, RuntimeError) as err:
        print(f'whole_run_ledger: {err}', file=sys.stderr)
        return 2
    print('\n'.join([f'{a.run} (core_bc {core_bc}, core_tfac {tfac:g})', *report(led)]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
