# Couple VULCAN chemistry to the AGNI climate of an atmosphere snapshot
from __future__ import annotations

import copy
import logging
import os
from typing import TYPE_CHECKING

import numpy as np

from proteus.atmos_chem.common import snapshot_star_file
from proteus.utils.helper import snapshot_path_for_time

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

# Loop count from which the AGNI energy solve uses its settings for a running evolution
LOOPS_SETTLED = 3

# Lowest temperature VULCAN is given [K], as for the uncoupled snapshot profile
VULCAN_TMP_FLOOR = 180.0


def _ensure_agni(dirs: dict, config: Config) -> None:
    """Load AGNI into Julia, unless this process has loaded it already (as `proteus start` has)."""
    from juliacall import Main as jl

    if not bool(jl.seval('isdefined(Main, :AGNI)')):
        from proteus.atmos_clim.agni import activate_julia

        activate_julia(dirs, config.atmos_clim.agni.verbosity)


class SnapshotClimate:
    """VULCAN climate hook that solves the climate of a PROTEUS atmosphere snapshot again.

    The AGNI atmosphere is set up as the evolution sets it up (`init_agni_atmos`), from the
    configuration and the helpfile row of the snapshot and with the stellar spectrum that
    snapshot used, and then loaded from the snapshot file, which restores its temperature,
    composition and surface state (magma temperature and conductive skin).

    Each call gives AGNI the mixing ratios of VULCAN (`vulcan.agni.set_composition`) and
    solves the climate through `run_agni`, the path of the evolution, with the surface
    condition of the snapshot and with AGNI's equilibrium chemistry switched off, so that the
    composition stays VULCAN's. Condensation (rainout) follows the configuration. The
    temperature and the eddy diffusion are returned in VULCAN's CGS units
    (`vulcan.agni.climate_update`), the temperature no lower than `VULCAN_TMP_FLOOR`.

    The result is not fed back to the interior: a postprocessed state is a diagnostic.

    Parameters
    ----------
    dirs : dict
        Directories of the run.
    config : Config
        Configuration of the run.
    hf_row : dict
        Helpfile row of the snapshot; `Time` [yr] selects the snapshot file.
    """

    def __init__(self, dirs: dict, config: Config, hf_row: dict):
        from juliacall import Main as jl

        from proteus.atmos_clim.agni import init_agni_atmos

        _ensure_agni(dirs, config)
        self.dirs = dirs
        self.hf_row = copy.copy(hf_row)  # keeps the report of a missing helpfile column
        self.time = float(hf_row['Time'])
        self.calls = 0

        # The solve uses VULCAN's composition; the setup keeps the gases of the snapshot
        self.config_solve = copy.deepcopy(config)
        self.config_solve.atmos_clim.agni.chemistry = None

        snapshot = snapshot_path_for_time(
            os.path.join(dirs['output'], 'data'), self.time, '_atm.nc'
        )
        self.atmos = init_agni_atmos(
            dirs,
            config,
            self.hf_row,
            use_cache=False,
            star_file=snapshot_star_file(dirs['output'], self.time),
            io_dir=os.path.join(dirs['output/offchem'], 'agni'),
        )
        if not bool(jl.AGNI.load.load_ncdf_b(self.atmos, snapshot)):
            self.close()
            raise RuntimeError(f'AGNI could not load the atmosphere snapshot {snapshot}')
        log.info('Climate coupled to the chemistry from snapshot %s' % snapshot)

    def __call__(self, pco, tco, species, ymix):
        """Solve the climate with VULCAN's composition; see `vulcan.climate`.

        Raises
        ------
        RuntimeError
            If AGNI does not converge, so that no unconverged climate reaches VULCAN.
        """
        from vulcan.agni import climate_update, set_composition

        from proteus.atmos_clim.agni import run_agni

        set_composition(self.atmos, pco, species, ymix)
        self.atmos, output = run_agni(
            self.atmos,
            LOOPS_SETTLED,
            self.dirs,
            self.config_solve,
            copy.copy(self.hf_row),
            False,
        )
        self.calls += 1
        if not output['agni_converged']:
            raise RuntimeError(f'AGNI did not converge at climate update {self.calls}')
        update = climate_update(self.atmos)
        update.tmp = np.maximum(update.tmp, VULCAN_TMP_FLOOR)
        return update

    def write(self):
        """Write the coupled AGNI state to ``data/<time>_chematm.nc``, beside the snapshot.

        Logs the net upward flux at the top of the atmosphere [W m-2] of the evolution at
        this row (`F_atm`) and of the coupled state; their difference is the change in energy
        balance that the chemistry makes.
        """
        from proteus.atmos_clim.agni import write_atmos_ncdf

        write_atmos_ncdf(self.atmos, self.dirs, self.time, suffix='_chematm.nc')
        f_coupled = float(np.asarray(self.atmos.flux_tot, dtype=float)[0])
        log.info(
            'Net flux at the top of the atmosphere: %.4e W m-2 in the evolution, '
            '%.4e W m-2 with the chemistry'
            % (float(self.hf_row.get('F_atm', np.nan)), f_coupled)
        )

    def close(self):
        """Deallocate the AGNI atmosphere."""
        from proteus.atmos_clim.agni import deallocate_atmos

        deallocate_atmos(self.atmos)
