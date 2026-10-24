#!/usr/bin/env python3

# Run PROTEUS for a grid of parameters

from __future__ import annotations

import gc
import itertools
import logging
import multiprocessing
import os
import shutil
import subprocess
import tempfile
import time
from copy import deepcopy
from datetime import datetime
from getpass import getuser
from numbers import Real

import numpy as np
import toml

from proteus.config import (
    Config,
    UnknownConfigKeyError,
    field_path_kind,
    read_config,
    read_config_object,
)
from proteus.config._interior import _TOL_UNSET, default_rtol, rtol_is_set
from proteus.utils.coupler import get_output_root
from proteus.utils.helper import get_proteus_dir, recursive_setattr
from proteus.utils.logs import setup_logger

PROTEUS_DIR = get_proteus_dir()


class _HeldRecords(logging.Handler):
    """Take the place of the fwl handlers and keep the records until manager.log exists."""

    def __init__(self, fwl: logging.Logger):
        super().__init__()
        self.records, self.saved = [], (fwl.handlers[:], fwl.level)

    def emit(self, record):
        self.records.append(record)


log = logging.getLogger('fwl.' + __name__)
_FWL = logging.getLogger('fwl')


def _output_root() -> str:
    """Return PROTEUS/output, the folder that holds the output of every grid."""
    return os.path.join(PROTEUS_DIR, 'output')


def _within(path: str, folder: str) -> bool:
    """Whether ``path`` is ``folder`` or lies below it, links resolved; folders are
    compared by identity, so a case variant on a case-insensitive file system matches."""
    try:
        target = os.stat(folder)
    except OSError:
        return False
    p = os.path.realpath(path)
    while True:
        try:
            if os.path.samestat(os.stat(p), target):
                return True
        except OSError:
            pass  # a part that cannot be stat'ed (missing or unreadable) is no match; go up
        if os.path.dirname(p) == p:
            return False
        p = os.path.dirname(p)


def _holds(folder: str, paths, resolved: bool = True) -> bool:
    """Whether removing ``folder`` takes one of ``paths``: by name, also through a link
    inside it, or with ``resolved`` by the file the path resolves to."""
    f = os.path.abspath(folder)
    return any(
        os.path.commonpath([os.path.abspath(p), f]) == f or (resolved and _within(p, folder))
        for p in paths
    )


def _check_symlink_dir(folder: str, inputs: tuple = ()):
    """Refuse a symlink folder that is a link, that is or lies in PROTEUS_DIR, the output
    folder (where output/ links to included) or the run output root (PROTEUS_OUTPUT_PATH),
    or that is or holds HOME, one of those or one of ``inputs`` (the grid file and base
    config, also through a link)."""
    if os.path.islink(folder):
        raise ValueError(f'symlink folder {folder} is a link: name the folder it points to')
    if os.path.exists(folder) and not os.path.isdir(folder):
        raise ValueError(f'symlink folder {folder} is not a folder')
    protected = (PROTEUS_DIR, _output_root(), get_output_root(PROTEUS_DIR))
    if (
        any(_within(folder, p) for p in protected)
        or any(_within(p, folder) for p in (*protected, os.path.expanduser('~')))
        or _holds(folder, inputs)
    ):
        raise ValueError(
            f'symlink folder {folder} (resolved {os.path.realpath(folder)}) lies in the '
            'PROTEUS or output folder, or holds HOME, one of them or a grid input: '
            'choose another folder'
        )


def _refuse_git(folder: str):
    """Refuse to empty a folder that holds a Git repository."""
    if any(f.name.lower() == '.git' for f in os.scandir(folder)):
        raise Exception('Not emptying directory - it contains a Git repository!')


def _replay(records: list):
    """Pass held records to the handlers the fwl logger has now, through the logger and
    handler levels as a live record would pass; not to root, which saw them live."""
    level = _FWL.getEffectiveLevel()
    for record in records:
        for handler in _FWL.handlers:
            if record.levelno >= max(level, handler.level):
                handler.handle(record)


# Thread target
def _thread_target(cfg_path, test, wait):
    if test:
        command = ['/bin/echo', 'Dummy output. Config file is at "' + cfg_path + '"']
    else:
        command = ['proteus', 'start', '--offline', '--config', cfg_path]
    subprocess.run(
        command, shell=False, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(wait)  # wait a bit, in case the process exited immediately


# Object for handling the parameter grid
class Grid:
    CONFIG_BASENAME = 'case_%06d'

    def __init__(
        self,
        name: str,
        base_config_path: str,
        symlink_dir: str = '_UNSET',
        grid_config: str = '',
        prepare_output: bool = True,
    ):
        """Set up the grid; with ``prepare_output=False`` the old output stays until
        `prepare_output` runs, and the fwl log lines are held until then: the caller must
        call `prepare_output` or `release_log`, or the fwl logger stays held.

        Raises
        ------
        ValueError
            If ``name`` resolves to PROTEUS/output itself or outside it (also through a
            link in output/) or names a folder that holds the grid file or the base
            config, if ``symlink_dir`` is not an absolute path, or if ``symlink_dir``
            would remove other data (see `_check_symlink_dir`).
        RuntimeError
            If another grid holds the log.
        Exception
            If the base config file does not exist or ``symlink_dir`` is blank.
        """
        # Grid's own name (for versioning, etc.)
        self.name = str(name).strip()
        self.outdir = PROTEUS_DIR + '/output/' + self.name + '/'
        self.conf = str(base_config_path)
        if not os.path.exists(self.conf):
            raise Exception("Base config file '%s' does not exist!" % self.conf)

        self.symlink_dir = symlink_dir
        if self.symlink_dir in ['', '.']:
            raise Exception('Symlinked directory is set to a blank path')
        self._inputs = tuple(p for p in (self.conf, grid_config) if p)
        if self.symlink_dir not in ('_UNSET', None):
            if not os.path.isabs(self.symlink_dir):
                raise ValueError(f'symlink folder {symlink_dir} must be an absolute path')
            self.symlink_dir = os.path.abspath(self.symlink_dir)  # the folder that is emptied
            _check_symlink_dir(self.symlink_dir, self._inputs)

        # Paths
        self.outdir = os.path.abspath(self.outdir) + '/'
        self._check_outdir()
        self.cfgdir = os.path.join(self.outdir, 'cfgs') + '/'
        self.logdir = os.path.join(self.outdir, 'logs') + '/'

        self.grid_config = grid_config
        self._held = None  # log records held until the output exists
        self._cfgs_adopted = False  # cfgs/ already holds the cases from adopt_cases

        # List of dimension names and parameter names
        self.dim_names = []  # String
        self.dim_param = []  # Dimension parameter of interest

        # Dimension variables (incl hypervariables)
        self.dim_avars = {}

        # Flattened Grid
        self.flat = []  # List of grid points, each is a dictionary
        self.size = 0  # Total size of grid

        if prepare_output:
            self.prepare_output()
        else:
            if any(isinstance(h, _HeldRecords) for h in _FWL.handlers):
                raise RuntimeError('another grid holds the log until its output exists')
            self._held = _HeldRecords(_FWL)
            _FWL.handlers = [self._held]
            _FWL.setLevel(logging.INFO)  # the level setup_logger sets; clears the level cache

    def _check_outdir(self):
        """Refuse a grid output that is PROTEUS/output itself or lies outside it, also
        through a link in output/, or that holds the grid file or the base config (a
        linked old output only by name, since it is only unlinked)."""
        root = os.path.abspath(_output_root())
        parent = os.path.dirname(self.outdir.rstrip('/'))
        if (
            os.path.commonpath([root, self.outdir]) != root
            or self.outdir == root + '/'
            or (os.path.exists(root) and not _within(parent, root))  # a link in the name
        ):
            raise ValueError(f"grid output '{self.name}' must name a folder inside {root}")
        out = self.outdir.rstrip('/')
        if _holds(out, self._inputs, resolved=not os.path.islink(out)):
            raise ValueError(
                f"grid output '{self.name}' holds the grid file or the base config: "
                'copy it out of the output folder first'
            )

    def _check_paths(self):
        """Check the grid output and the symlink folder against the folders they may not
        remove."""
        self._check_outdir()
        if self.symlink_dir not in ('_UNSET', None):
            _check_symlink_dir(self.symlink_dir, self._inputs)

    def _empty(self, folder: str):
        """Remove ``folder`` after a wait, with the path checks run again right before."""
        print("Removing old files at '%s'" % folder)
        time.sleep(4.0)
        self._check_paths()
        _refuse_git(folder)
        shutil.rmtree(folder)

    def prepare_output(self):
        """Remove the old output, create the output folders, copy the configs and start
        manager.log, then log there the lines held since the grid was set up."""
        self._check_paths()

        # Remove old output location; a trailing slash would follow an old output link
        out = self.outdir.rstrip('/')
        if os.path.islink(out):
            os.unlink(out)
        if os.path.isdir(out):
            self._empty(out)

        # Create new output location
        if self.symlink_dir in ['_UNSET', None]:
            # Not using symlink
            self.using_symlink = False
            os.makedirs(self.outdir)
        else:
            # Will be using symlink
            self.using_symlink = True
            if os.path.exists(self.symlink_dir):
                self._empty(self.symlink_dir)
            os.makedirs(self.symlink_dir)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            os.symlink(self.symlink_dir, out)

        # Make subdirectories
        for dir in [self.cfgdir, self.logdir]:
            if os.path.exists(dir):
                shutil.rmtree(dir)
            os.makedirs(dir)

        # Make copy of GRID config file, if there is one
        if self.grid_config:
            shutil.copyfile(self.grid_config, os.path.join(self.outdir, 'copy.grid.toml'))

        # Make copy of REFERENCE config file
        shutil.copyfile(self.conf, os.path.join(self.outdir, 'ref_config.toml'))

        # Setup logging. Keep the plain timestamped layout the grid manager has
        # always written to manager.log, via the shared setup_logger.
        setup_logger(
            logpath=os.path.join(self.outdir, 'manager.log'),
            logterm=True,
            level='INFO',
            fmt='[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

        held = self._unhold_log()
        log.info("Grid '%s' initialised empty" % self.name)
        _replay(held)

    def release_log(self):
        """Stop holding log lines and pass the held ones on, once."""
        _replay(self._unhold_log())

    def _unhold_log(self) -> list:
        """Stop holding log lines; return the held records (none once released)."""
        held, self._held = self._held, None
        if held is None:
            return []
        if held in _FWL.handlers:  # setup_logger has not replaced them
            _FWL.handlers = held.saved[0]
            _FWL.setLevel(held.saved[1])
        return held.records

    # Add a new empty dimension to the Grid
    def add_dimension(self, name: str, var: str):
        if name in self.dim_names:
            raise Exception("Dimension '%s' cannot be added twice" % name)
        if var in (
            'interior_energetics.num_tolerance',
            'interior_energetics.spider.tolerance_rel',
        ):
            raise ValueError(
                f'{var} is a deprecated alias; sweep interior_energetics.rtol instead'
            )

        log.info("Added new dimension '%s' " % name)
        log.debug(
            "Dimension '%s' maps to parameter '%s' (total dims: %d)",
            name,
            var,
            len(self.dim_names) + 1,
        )
        self.dim_names.append(name)
        self.dim_param.append(var)
        self.dim_avars[name] = None

    def _get_idx(self, name: str):
        if name not in self.dim_names:
            raise Exception("Dimension '%s' is not initialised" % name)
        return int(self.dim_names.index(name))

    # Set a dimension by linspace
    def set_dimension_linspace(self, name: str, start: float, stop: float, count: int):
        self.dim_avars[name] = list(np.linspace(start, stop, count))
        self.dim_avars[name] = [float(v) for v in self.dim_avars[name]]
        log.debug(
            "Dimension '%s' linspace: %d points in [%.6g, %.6g]", name, count, start, stop
        )

    # Set a dimension by arange (inclusive of endpoint)
    def set_dimension_arange(self, name: str, start: float, stop: float, step: float):
        self.dim_avars[name] = list(np.arange(start, stop, step))
        if not np.isclose(self.dim_avars[name][-1], stop):
            self.dim_avars[name].append(stop)
        self.dim_avars[name] = [float(v) for v in self.dim_avars[name]]
        log.debug(
            "Dimension '%s' arange: %d points in [%.6g, %.6g] (step=%.6g)",
            name,
            len(self.dim_avars[name]),
            start,
            stop,
            step,
        )

    # Set a dimension by logspace
    def set_dimension_logspace(self, name: str, start: float, stop: float, count: int):
        self.dim_avars[name] = list(np.logspace(np.log10(start), np.log10(stop), count))
        self.dim_avars[name] = [float(v) for v in self.dim_avars[name]]
        log.debug(
            "Dimension '%s' logspace: %d points in [%.6g, %.6g]", name, count, start, stop
        )

    # Set a dimension directly
    def set_dimension_direct(self, name: str, values: list, sort: bool = True):
        why = direct_values_error(values)
        if why:
            raise ValueError(f"direct dimension '{name}' {why}")
        values = [v.item() if isinstance(v, np.generic) else v for v in values]  # array, tuple
        the_list = list(dict.fromkeys(values))  # written order, duplicates removed
        if (not isinstance(values[0], str)) and sort:  # sort if numeric
            the_list = sorted(the_list)
        self.dim_avars[name] = the_list
        log.debug(
            "Dimension '%s' direct: %d values (deduplicated from %d)",
            name,
            len(the_list),
            len(values),
        )

    # Print current setup
    def print_setup(self):
        log.info('Grid configuration')
        log.info(' -- name     : %s' % self.name)
        log.info(' ')
        for name in self.dim_names:
            idx = self._get_idx(name)

            log.info(' -- dimension: %s' % name)
            log.info('    parameter: %s' % self.dim_param[idx])
            log.info('    values   : %s' % self.dim_avars[name])
            log.info('    length   : %d' % len(self.dim_avars[name]))
            log.info(' ')

    # Print generated grid
    def print_grid(self):
        log.info('Flattened grid points')
        for i, gp in enumerate(self.flat):
            log.info('    %d: %s' % (i, gp))
        log.info(' ')

    def _get_tmpcfg(self, idx: int):
        return os.path.join(self.cfgdir, str(self.CONFIG_BASENAME % idx) + '.toml')

    # Generate the Grid based on the current configuration
    def generate(self):
        log.info('Generating parameter grid')

        # Validate
        if len(self.dim_avars) == 0:
            raise Exception('Grid is empty')

        # Dimension count
        log.info('    %d dimensions' % len(self.dim_names))

        # Calculate total Grid size
        values = list(self.dim_avars.values())
        self.size = 1
        for v in values:
            self.size *= len(v)
        log.info('    %d points expected' % self.size)
        log.debug(
            'Grid size breakdown: %s',
            ' x '.join('%s(%d)' % (n, len(self.dim_avars[n])) for n in self.dim_names),
        )

        # Create flattened parameter grid
        flat_values = list(itertools.product(*values))
        log.info('    created %d grid points' % len(flat_values))

        # Re-assign keys to values
        log.info('    mapping keys')
        for fl in flat_values:
            gp = {}

            for i in range(len(fl)):
                gp[self.dim_param[i]] = fl[i]

            self.flat.append(gp)

        self.size = len(self.flat)
        log.info('    done')
        log.info(' ')

    def write_config_files(self):
        """Write the config file of each case into cfgs/ and read it back with the config
        loader, unless `adopt_cases` has already placed the checked ones there.

        Raises
        ------
        ValueError
            As `write_cases`.
        """
        if not self._cfgs_adopted:
            self.write_cases(self.cfgdir)

    def adopt_cases(self, tmpdir: str):
        """Move the case configs checked in ``tmpdir`` into the output's cfgs folder."""
        for name in os.listdir(tmpdir):
            shutil.move(os.path.join(tmpdir, name), os.path.join(self.cfgdir, name))
        self._cfgs_adopted = True

    def write_cases(self, folder: str):
        """Write each case config into ``folder`` and read it back with the config loader.

        Raises
        ------
        ValueError
            If a value is refused when it is set, as 'Grid case i (key = value): reason',
            or if a written case does not load, as 'Grid case i (key = value, ...): ' and
            the loader's message, which names the case file without its folder.
        """
        log.info('Writing config files')
        base_config = read_config_object(self.conf)
        # A module set per case takes that module's default rtol when nothing sets one.
        rtol_unset = not rtol_is_set(read_config(self.conf).get('interior_energetics', {}))
        for i, gp in enumerate(self.flat):
            thisconf: Config = deepcopy(base_config)
            thisconf.params.out.path = self.name + '/' + str(self.CONFIG_BASENAME % i) + '/'
            for key in gp.keys():
                try:
                    recursive_setattr(thisconf, key, gp[key])
                except (TypeError, ValueError) as exc:
                    reason = exc.args[0] if exc.args and isinstance(exc.args[0], str) else exc
                    raise ValueError(f'Grid case {i} ({key} = {gp[key]!r}): {reason}') from None
            if 'interior_energetics.rtol' in gp:  # a base alias would conflict on reload
                ie = thisconf.interior_energetics
                ie.num_tolerance = ie.spider.tolerance_rel = _TOL_UNSET
            elif rtol_unset and 'interior_energetics.module' in gp:
                ie = thisconf.interior_energetics
                ie.rtol = default_rtol(ie.module)

            # Write this configuration file and read it back, so an invalid case stops here
            case_path = os.path.join(folder, os.path.basename(self._get_tmpcfg(i)))
            thisconf.write(case_path)
            os.sync()
            try:
                read_config_object(case_path)
            except ValueError as exc:
                where = ', '.join(f'{k} = {v!r}' for k, v in gp.items())
                reason = str(exc).replace(case_path, os.path.basename(case_path))
                raise ValueError(f'Grid case {i} ({where}): {reason}') from None

    def run(
        self,
        num_threads: int,
        test_run: bool = False,
        check_interval: float = 15.0,
        print_interval: float = 8,
    ):
        """
        Run GridPROTEUS on the current machine.

        The value of `num_threads` will be restricted to the number of available CPUs.

        Parameters
        ----------
        - `num_threads:int`         number of CPU cores to use
        - `test_run:bool`           if true, does not actually run PROTEUS
        - `check_interval:float`    interval [secs] at which to check status of workers
        - `print_interval:int`      step interval at which to print (8*15 seconds = 2 minutes)
        """

        log.info("Running PROTEUS across parameter grid '%s'" % self.name)

        time_start = datetime.now()
        log.info("Output path: '%s'" % self.outdir)
        if self.using_symlink:
            log.info("Symlink target: '%s'" % self.symlink_dir)
        # do not need more threads than there are points
        num_threads = min(num_threads, self.size)

        # do not use more threads than are available
        num_threads = min(num_threads, os.cpu_count())
        log.debug(
            'Thread pool: %d threads (grid_size=%d, cpu_count=%d)',
            num_threads,
            self.size,
            os.cpu_count(),
        )

        # Print warning
        if not test_run:
            log.info(' ')
            log.info('Confirm that the grid above is what you had intended.')
            log.info(
                'There are %d grid points. There will be %d threads.' % (self.size, num_threads)
            )
            log.info(
                'Do not close this program! It must stay alive to manage each of the subproceses.'
            )
            log.info(' ')

            log.info('Sleeping...')
            for i in range(7, 0, -1):
                log.info('    %d ' % i)
                time.sleep(1.0)
            log.info(' ')

        # Print more often if this is a test
        else:
            check_interval = 1.0
            print_interval = 1

        self.write_config_files()

        gc.collect()

        # Setup threads
        threads = []
        for i in range(self.size):
            cfg_path = self._get_tmpcfg(i)

            # Check cfg exists
            cfgexists = False
            waitfor = 3.0
            waitcount = 100.0
            for _ in range(int(waitfor * waitcount)):  # do N=waitcount*waitfor checks
                time.sleep(1 / waitcount)
                if os.path.exists(cfg_path):
                    cfgexists = True
                    break
            if not cfgexists:
                raise Exception('Config file could not be found for case %d!' % i)

            # Add process
            threads.append(
                multiprocessing.Process(
                    target=_thread_target, args=(cfg_path, test_run, check_interval * 3)
                )
            )

        # Track statuses
        # 0: queued
        # 1: running
        # 2: done
        status = np.zeros(self.size)
        done = False  # All done?
        log.info(
            'Starting process manager (%d grid points, %d threads)' % (self.size, num_threads)
        )
        step = 0
        while not done:
            # Check alive
            n_run = 0
            for i in range(self.size):
                # if not marked as running, don't do anything here
                if status[i] != 1:
                    continue

                # marked as running
                if threads[i].is_alive():
                    # it is still going
                    status[i] = 1
                    n_run += 1
                else:
                    # it has completed
                    if status[i] == 1:
                        status[i] = 2
                        threads[i].join()  # make everything wait until it's completed
                        threads[i].close()  # release resources
                        gc.collect()

            # Print info
            count_que = np.count_nonzero(status == 0)
            count_run = np.count_nonzero(status == 1)
            count_end = np.count_nonzero(status == 2)
            if step % print_interval == 0:
                log.info(
                    '%3d queued (%5.1f%%), %3d running (%5.1f%%), %3d exited (%5.1f%%)'
                    % (
                        count_que,
                        100.0 * count_que / self.size,
                        count_run,
                        100.0 * count_run / self.size,
                        count_end,
                        100.0 * count_end / self.size,
                    )
                )

            # Done?
            if np.count_nonzero(status == 2) == self.size:
                done = True
                log.info('All cases have exited')
                break

            # Start new?
            start_new = np.count_nonzero(status == 1) < num_threads
            if start_new:
                for i in range(self.size):
                    if status[i] == 0:
                        status[i] = 1
                        start_new = False
                        log.debug('Dispatching case %06d (step %d)', i, step)
                        threads[i].start()
                        break

            # Short sleeps while doing initial dispatch
            if step < num_threads:
                time.sleep(1.0)
            else:
                time.sleep(check_interval)
            step += 1
            # / end while loop

        # Check all cases' status files
        if not test_run:
            for i in range(self.size):
                # find file
                status_path = os.path.join(self.outdir, self.CONFIG_BASENAME % i, 'status')
                if not os.path.exists(status_path):
                    raise Exception("Cannot find status file at '%s'" % status_path)

                # read file
                with open(status_path, 'r') as hdl:
                    lines = hdl.readlines()
                this_stat = int(lines[0])

                # if still marked as running, it must have died at some point
                if 0 <= this_stat <= 9:
                    log.warning(
                        'Case %06d has status=running but it is not alive. Setting status=died.'
                        % i
                    )
                    with open(status_path, 'w') as hdl:
                        hdl.write('25\n')
                        hdl.write('Error (died)\n')

        time_end = datetime.now()
        log.info('All processes finished at: ' + str(time_end.strftime('%Y-%m-%d_%H:%M:%S')))
        log.info(
            'Total runtime: %.1f hours ' % ((time_end - time_start).total_seconds() / 3600.0)
        )

    def slurm_config(
        self,
        max_jobs: int,
        test_run: bool = False,
        max_days: int = 1,
        max_mem: int = 12,
        jax_cache: bool = False,
    ):
        """Write slurm config file.

        Uses a slurm job array, see link for more info:
        https://slurm.schedmd.com/job_array.html

        Parameters
        ----------
        max_jobs : int
            Maximum number of jobs to run
        test_run : bool
            If true, generate dummy commands to test the code.
        max_days : int
            Maximum number of days to run
        max_mem : int
            Maximum memory per CPU in GB
        jax_cache : bool
            If true, the generated script enables a bounded JAX persistent
            compilation cache shared across the array tasks (stored under the
            grid output directory) so compiled executables are reused and JIT
            recompiles are cut. An LRU size bound caps the cache so it cannot
            fill the filesystem. Default false: no cache environment variables
            are written and the script behaves exactly as before.
        """

        max_days = int(max_days)  # ensure integer
        max_jobs = min(max_jobs, self.size)  # do not request more jobs than we need

        log.info("Generating PROTEUS slurm config for parameter grid '%s'" % self.name)
        log.info('Will run for up to %d days per job' % max_days)
        log.info('Will use up to %d GB of memory per job' % max_mem)
        log.info(' ')

        log.info("Output path: '%s'" % self.outdir)
        if self.using_symlink:
            log.info("Symlink target: '%s'" % self.symlink_dir)

        log.info(' ')
        self.write_config_files()

        if test_run:
            command = '/bin/echo Dummy output. Config file is at: '
        else:
            command = 'proteus start --offline --config'

        log_file = os.path.join(self.logdir, 'proteus-%A_%a.log')

        # Optional bounded JAX persistent compilation cache. The size bound uses
        # JAX's own LRU eviction so the cache plateaus instead of growing without
        # limit; the min-compile-time and min-entry-size thresholds keep trivial
        # compilations out of it. Disabled by default so the generated script is
        # unchanged unless the cache is explicitly requested.
        if jax_cache:
            jax_cache_dir = os.path.join(self.outdir, 'jax_cache')
            jax_env = (
                f'export JAX_COMPILATION_CACHE_DIR={jax_cache_dir}\n'
                'export JAX_COMPILATION_CACHE_MAX_SIZE=85899345920\n'
                'export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1\n'
                'export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=4096\n'
            )
        else:
            jax_env = ''

        string = f"""#!/bin/sh
#SBATCH -J proteus.grid.array
#SBATCH --export=ALL
#SBATCH --time={max_days}-00
#SBATCH --mem-per-cpu={max_mem}G
#SBATCH -i /dev/null
#SBATCH -o {log_file}
#SBATCH --array=0-{self.size - 1}%{max_jobs}

{jax_env}i=$SLURM_ARRAY_TASK_ID

while [ $i -lt {self.size} ]; do
    printf -v cfg "{self.cfgdir}/{self.CONFIG_BASENAME}.toml" $((i))
    echo executing proteus with config $cfg
    {command} $cfg
    i=$((i+{self.size}))
done

"""

        slurm_path = os.path.join(self.outdir, 'slurm_dispatch.sh')
        with open(slurm_path, 'w') as f:
            f.write(string)

        user = getuser()
        log.info('')
        log.info('Submit to Slurm using:')
        log.info('    `sbatch %s`', slurm_path)
        log.info('')
        log.info('To look at the Slurm queue:')
        log.info('    `squeue` or `squeue -u %s`', user)
        log.info('')
        log.info('To cancel a job, or all of your jobs:')
        log.info('    `scancel [jobid]`, or `scancel -u %s`', user)

        time.sleep(2.0)


# Top-level keys of a grid file that set up the grid; every other key names a dimension.
GRID_SETTINGS = frozenset(
    {
        'output',
        'ref_config',
        'symlink',
        'use_slurm',
        'max_jobs',
        'max_days',
        'max_mem',
        'jax_cache',
        'config_version',
    }
)


def direct_values_error(values) -> str | None:
    """Return why a direct grid axis cannot be built from ``values``, or None if it can.

    The axis needs a non-empty list, tuple or 1-D array of numbers and booleans, or of
    strings: duplicates are removed and numbers sorted, which has no defined result for a
    NaN (it equals nothing, itself included) or for strings beside other values.
    """
    if not (
        isinstance(values, (list, tuple))
        or (isinstance(values, np.ndarray) and values.ndim == 1)
    ):
        return f'is not a list of values: {values!r}'
    values = list(values)
    if not values:
        return 'has no values'
    for v in values:
        if not isinstance(v, (Real, np.bool_, str)):
            return f'holds {v!r}, which is not a real number, boolean or string'
    if any(v != v for v in values):
        return 'holds NaN, which cannot be deduplicated or ordered'
    first = isinstance(values[0], str)
    other = next((v for v in values if isinstance(v, str) != first), None)
    if other is not None:
        return f'mixes strings with other values ({values[0]!r} and {other!r})'
    return None


# Keys each dimension method reads from its table, in the order of its Grid setter.
DIMENSION_METHODS = {
    'direct': ('values',),
    'linspace': ('start', 'stop', 'count'),
    'logspace': ('start', 'stop', 'count'),
    'arange': ('start', 'stop', 'step'),
}


def _dimension_kind(key: str, value) -> str:
    """Classify a non-setting grid key; ``'field'`` means a complete dimension table."""
    kind = field_path_kind(key)
    if kind != 'field':
        return kind
    if not (isinstance(value, dict) and 'method' in value):
        return 'value'
    method = value['method']
    if not isinstance(method, str) or method not in DIMENSION_METHODS:
        return 'method'
    return 'field' if all(arg in value for arg in DIMENSION_METHODS[method]) else 'incomplete'


def grid_dimension_keys(config: dict, config_fpath: str) -> list[str]:
    """Return the dimension keys of a grid config, refusing any key it cannot account for.

    A top-level key is either one of ``GRID_SETTINGS`` or a dimension: the path of a
    single configuration field in the PROTEUS schema, given as a table with one of the
    ``DIMENSION_METHODS`` and the keys that method reads. Whether a key is a field is
    decided by the schema, not by a dot.

    Parameters
    ----------
    config:
        The grid config as parsed from TOML.
    config_fpath:
        Path of the grid file, quoted back in the error.

    Returns
    -------
    list[str]
        The dimension keys, in file order.

    Raises
    ------
    UnknownConfigKeyError
        If a key is neither a grid setting nor a schema field, names a whole section
        rather than one field, or does not hold a table with a known ``method`` and the
        keys it reads.
    """
    dims = [key for key in config if key not in GRID_SETTINGS]
    kinds = {key: _dimension_kind(key, config[key]) for key in dims}
    reasons = {
        'unknown': 'not a grid setting or a configuration field',
        'section': 'a configuration section, not a single field (quote a dotted table '
        'header, as in ["planet.mass_tot"])',
        'value': 'a configuration field without a dimension table holding a method',
        'method': f'a dimension table whose method is not one of {", ".join(DIMENSION_METHODS)}',
        'incomplete': 'a dimension table without a key its method reads ('
        + '; '.join(f'{m}: {", ".join(args)}' for m, args in DIMENSION_METHODS.items())
        + ')',
    }
    lines = []
    for kind, reason in reasons.items():
        keys = ', '.join(f'"{key}"' for key in dims if kinds[key] == kind)
        if keys:
            lines.append(f'  {reason}: {keys}')
    if lines:
        lines.insert(0, f'Grid config {config_fpath} has keys the grid cannot account for:')
        lines.append(
            f'  A key is a grid setting ({", ".join(sorted(GRID_SETTINGS))}) or the path '
            'of one configuration field to sweep, such as "planet.mass_tot".'
        )
        raise UnknownConfigKeyError('\n'.join(lines))
    return dims


def _grid_bool(value, key: str, fpath: str) -> bool:
    """Return a grid setting that must be a TOML true or false; refuse any other value."""
    if not isinstance(value, bool):
        raise ValueError(
            f'grid file {fpath}: grid setting {key} must be true or false, got {value!r}'
        )
    return value


def grid_from_config(config_fpath: str, test_run: bool = False, check_interval: float = 15.0):
    """Run GridPROTEUS using the parameters in a config file (xxx.grid.toml)"""

    # Load configuration from TOML file
    with open(config_fpath, 'r') as file:
        config = toml.load(file)
    dimension_keys = grid_dimension_keys(config, config_fpath)
    for key in dimension_keys:
        why = config[key]['method'] == 'direct' and direct_values_error(config[key]['values'])
        if why:
            raise ValueError(f'grid file {config_fpath}: the direct axis "{key}" {why}')

    # Output folder name, created inside `PROTEUS/output/`
    folder = str(config['output'])
    if folder.strip().lower() == 'auto':
        import secrets
        from datetime import datetime

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        suffix = secrets.token_hex(2)
        folder = f'grid_{stamp}_{suffix}'

    # Set this string to have the output files created at an alternative location. The
    #   output 'folder' in `PROTEUS/output/` will then by symbolically linked to this
    #   alternative location. Useful for when data should be saved on a storage server.
    #   THIS MUST BE AN ABSOLUTE PATH
    symlink = str(config['symlink'])
    if symlink.strip().lower() in ('', 'none', 'false'):
        symlink = '_UNSET'
    else:
        if not os.path.isabs(symlink):
            raise RuntimeError('Path for symbolic link must be absolute.\n\tGot: ' + symlink)

    # Use SLURM?
    if 'use_slurm' not in config:
        raise KeyError(f'grid file {config_fpath}: the grid setting use_slurm is missing')
    use_slurm = _grid_bool(config['use_slurm'], 'use_slurm', config_fpath)

    # Execution limits
    max_jobs = int(config['max_jobs'])  # maximum number of concurrent tasks (e.g. 300)
    max_days = int(config['max_days'])  # maximum number of days to run (e.g. 1)
    max_mem = int(config['max_mem'])  # maximum memory per CPU in GB (e.g. 3)

    # Optional bounded JAX persistent compilation cache (Slurm runs only).
    # Absent or false leaves the generated script cache-free.
    jax_cache = _grid_bool(config.get('jax_cache', False), 'jax_cache', config_fpath)

    # Base config file
    cfg_base = os.path.join(PROTEUS_DIR, str(config['ref_config']))

    # Set up the grid, and check every case config before the old output is removed
    pg = Grid(
        folder, cfg_base, symlink_dir=symlink, grid_config=config_fpath, prepare_output=False
    )
    try:
        # Add dimensions to grid by looping over keys
        for dim, key in enumerate(dimension_keys):
            name = 'param_%03d' % dim
            pg.add_dimension(name, key)

            table = config[key]
            setter = getattr(pg, f'set_dimension_{table["method"]}')
            try:
                setter(name, *(table[arg] for arg in DIMENSION_METHODS[table['method']]))
            except (ValueError, ZeroDivisionError) as exc:
                raise ValueError(f'grid dimension {key} ({table["method"]}): {exc}') from None

        # Print information
        pg.print_setup()
        pg.generate()

        # Sanity check grid size (cannot allow larger values due to format of file paths)
        if pg.size > 999999:
            raise ValueError('Number of grid points is too large')

        # Print each grid point if the number of points isn't too large
        if pg.size < 1e4:
            pg.print_grid()
        else:
            log.warning('Number of grid points exceeds 10000. Will not print them here.')

        os.makedirs(_output_root(), exist_ok=True)
        with tempfile.TemporaryDirectory(dir=_output_root(), prefix='.grid-check-') as tmpdir:
            pg.write_cases(tmpdir)
            pg.prepare_output()
            pg.adopt_cases(tmpdir)
    finally:
        pg.release_log()

    # Run the grid
    if use_slurm:
        # Generate Slurm batch file, use `sbatch` to submit
        pg.slurm_config(
            max_jobs,
            test_run=test_run,
            max_days=max_days,
            max_mem=max_mem,
            jax_cache=jax_cache,
        )
    else:
        # Alternatively, let grid_proteus.py manage the jobs
        pg.run(max_jobs, test_run=test_run, check_interval=check_interval)
        log.info('GridPROTEUS finished')
