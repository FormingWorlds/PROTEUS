# Generic atmospheric chemistry wrapper
from __future__ import annotations

import functools
import logging
import os
import shutil
from typing import TYPE_CHECKING

import pandas as pd

from proteus.atmos_chem.common import (
    chem_csv_name,
    helpfile_row_at,
    read_result,
    resolve_chemistry_times,
)
from proteus.utils.helper import UpdateStatusfile

log = logging.getLogger('fwl.' + __name__)

if TYPE_CHECKING:
    from proteus.config import Config


def run_chemistry(dirs: dict, config: Config, hf_all: pd.DataFrame) -> pd.DataFrame | None:
    """
    Post-process atmosphere snapshots with the atmospheric chemistry model.

    Runs at the snapshot times set by `atmos_chem.times`, in ascending order,
    after the simulation. A snapshot without an atmosphere (surface pressure
    zero, as after desiccation) is left out. Each run writes
    ``offchem/<module>_<time>.csv``, and the latest is also copied to
    ``offchem/<module>.csv``. The ``offchem`` folder is cleared once at the
    start. A failed run stops the sequence; the results before it are kept.

    Parameters
    ----------
    dirs : dict
        Dictionary of directories.
    config : Config
        Configuration object.
    hf_all : pd.DataFrame
        Helpfile of the run. Each run uses the row nearest its snapshot time.

    Returns
    ----------
        result : pd.DataFrame | None
            Result of the latest run, or None if there was nothing to run.

    Raises
    ----------
        RuntimeError
            If the chemistry model fails at a snapshot.
    """

    log.info('Running atmospheric chemistry...')

    # The config converter maps the TOML string 'none' to None
    module = config.atmos_chem.module
    if not module:
        log.warning('Cannot run atmospheric chemistry, no module specified')
        return None

    # Resolve the runner function (lazy imports keep heavy backends optional)
    if module == 'vulcan':
        from proteus.atmos_chem.vulcan import run_vulcan

        # The snapshots of this call share the state of the network compilation
        _run = functools.partial(run_vulcan, network_state={})
    elif module == 'dummy':
        from proteus.atmos_chem.dummy import run_dummy_chem as _run
    else:
        if dirs.get('output'):
            UpdateStatusfile(dirs, 20)
        raise ValueError(f"Invalid atmos_chem module: '{module}'")

    times = resolve_chemistry_times(dirs['output'], hf_all, config.atmos_chem.times)
    no_atm = [t for t in times if not float(helpfile_row_at(hf_all, t)['P_surf']) > 0.0]
    if no_atm:
        log.warning(
            'Skipping %d snapshot(s) without an atmosphere (desiccated), from t=%.3e yr'
            % (len(no_atm), no_atm[0])
        )
        times = [t for t in times if t not in no_atm]
    if not times:
        log.warning('No snapshots to run atmospheric chemistry on')
        return None
    log.info(f'    Using {module} module at {len(times)} snapshot(s)')

    shutil.rmtree(dirs['output/offchem'], ignore_errors=True)
    os.makedirs(dirs['output/offchem'])

    # Rows report a column absent from an old helpfile, rather than a KeyError
    from proteus.utils.coupler import HelpfileRow, helpfile_path

    for i, time in enumerate(times):
        log.info('    Chemistry snapshot %d of %d, t=%.3e yr' % (i + 1, len(times), time))
        # Snapshot file names hold the time to 1e-3 yr, so the row takes that time
        snapshot = helpfile_row_at(hf_all, time)
        snapshot['Time'] = time
        hf_row = HelpfileRow(snapshot, helpfile_path(dirs['output']))
        if not _run(dirs, config, hf_row):
            raise RuntimeError('%s chemistry failed at t=%.3e yr' % (module, time))

    # The fixed name holds the latest result
    offchem = dirs['output/offchem']
    shutil.copyfile(
        os.path.join(offchem, chem_csv_name(module, times[-1])),
        os.path.join(offchem, f'{module}.csv'),
    )
    return read_result(dirs['output'], module)
