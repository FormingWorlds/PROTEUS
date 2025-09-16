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
import sys
import time
from copy import deepcopy
from datetime import datetime
from getpass import getuser

import numpy as np
import toml

from proteus.config import Config, read_config_object
from proteus.utils.helper import get_proteus_dir, recursive_setattr

PROTEUS_DIR=get_proteus_dir()

# Custom logger instance
def setup_logger(logpath:str="new.log",level=1,logterm=True):

    # https://stackoverflow.com/a/61457119

    custom_logger = logging.getLogger()
    custom_logger.handlers.clear()

    if os.path.exists(logpath):
        os.remove(logpath)

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt='%Y-%m-%d %H:%M:%S')

    level_code = logging.INFO
    match level:
        case 0:
            level_code = logging.DEBUG
        case 2:
            level_code = logging.WARNING
        case 3:
            level_code = logging.ERROR
        case 4:
            level_code = logging.CRITICAL

    # Add terminal output to logger
    if logterm:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh.setLevel(level_code)
        custom_logger.addHandler(sh)

    # Add file output to logger
    fh = logging.FileHandler(logpath)
    fh.setFormatter(fmt)
    fh.setLevel(level)
    custom_logger.addHandler(fh)
    custom_logger.setLevel(level_code)

    # Capture unhandled exceptions
    # https://stackoverflow.com/a/16993115
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            custom_logger.error("KeyboardInterrupt")
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        custom_logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = handle_exception

    return

# Thread target
def _thread_target(cfg_path, test, wait):
    if test:
        command = ['/bin/echo','Dummy output. Config file is at "' + cfg_path + '"']
    else:
        command = ["proteus","start","--offline","--config",cfg_path]
    subprocess.run(command, shell=False, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(wait)  # wait a bit, in case the process exited immediately

# Object for handling the parameter grid
class Grid():

    CONFIG_BASENAME = "case_%06d"

    def __init__(self, name:str, base_config_path:str,
                 symlink_dir:str="_UNSET",
                 grid_config:str=""):

        # Grid's own name (for versioning, etc.)
        self.name = str(name).strip()
        self.outdir = PROTEUS_DIR+"/output/"+self.name+"/"
        self.conf = str(base_config_path)
        if not os.path.exists(self.conf):
            raise Exception("Base config file '%s' does not exist!" % self.conf)

        self.symlink_dir = symlink_dir
        if self.symlink_dir in ["","."]:
            raise Exception("Symlinked directory is set to a blank path")

        # Paths
        self.outdir = os.path.abspath(self.outdir) + "/"
        self.cfgdir = os.path.join(self.outdir, "cfgs") + "/"
        self.logdir = os.path.join(self.outdir, "logs") + "/"

        # Remove old output location
        if os.path.exists(self.outdir):
            if os.path.islink(self.outdir):
                os.unlink(self.outdir)
            if os.path.isdir(self.outdir):
                print("Removing old files at '%s'" % self.outdir)
                subfolders = [ f.path.split("/")[-1].lower() for f in os.scandir(self.outdir) if f.is_dir() ]
                if ".git" in subfolders:
                    raise Exception("Not emptying directory - it contains a Git repository!")
                time.sleep(4.0)
                shutil.rmtree(self.outdir)

        # Create new output location
        if self.symlink_dir in ["_UNSET", None]:
            # Not using symlink
            self.using_symlink = False
            os.makedirs(self.outdir)
        else:
            # Will be using symlink
            self.using_symlink = True
            self.symlink_dir = os.path.abspath(self.symlink_dir)
            if os.path.exists(self.symlink_dir):
                print("Removing old files at '%s'" % self.symlink_dir)
                subfolders = [ f.path.split("/")[-1].lower() for f in os.scandir(self.symlink_dir) if f.is_dir() ]
                if ".git" in subfolders:
                    raise Exception("Not emptying directory - it contains a Git repository!")
                time.sleep(4.0)
                shutil.rmtree(self.symlink_dir)
            os.makedirs(self.symlink_dir)
            os.symlink(self.symlink_dir, self.outdir)

        # Make subdirectories
        for dir in [self.cfgdir, self.logdir]:
            if os.path.exists(dir):
                shutil.rmtree(dir)
            os.makedirs(dir)

        # Make copy of GRID config file, if there is one
        if grid_config:
            shutil.copyfile(grid_config, os.path.join(self.outdir, "copy.grid.toml"))

        # Make copy of REFERENCE config file
        shutil.copyfile(self.conf, os.path.join(self.outdir, "ref_config.toml"))

        # Setup logging
        setup_logger(logpath=os.path.join(self.outdir,"manager.log"), logterm=True, level=1)
        global log
        log = logging.getLogger(__name__)

        log.info("Grid '%s' initialised empty" % self.name)

        # List of dimension names and parameter names
        self.dim_names = []     # String
        self.dim_param = []     # Dimension parameter of interest

        # Dimension variables (incl hypervariables)
        self.dim_avars = {}

        # Flattened Grid
        self.flat = []   # List of grid points, each is a dictionary
        self.size = 0    # Total size of grid

    # Add a new empty dimension to the Grid
    def add_dimension(self,name:str,var:str):
        if name in self.dim_names:
            raise Exception("Dimension '%s' cannot be added twice" % name)

        log.info("Added new dimension '%s' " % name)
        self.dim_names.append(name)
        self.dim_param.append(var)
        self.dim_avars[name] = None

    def _get_idx(self,name:str):
        if name not in self.dim_names:
            raise Exception("Dimension '%s' is not initialised" % name)
        return int(self.dim_names.index(name))

    # Set a dimension by linspace
    def set_dimension_linspace(self,name:str,start:float,stop:float,count:int):
        self.dim_avars[name] = list(np.linspace(start,stop,count))
        self.dim_avars[name] = [float(v) for v in self.dim_avars[name]]

    # Set a dimension by arange (inclusive of endpoint)
    def set_dimension_arange(self,name:str,start:float,stop:float,step:float):
        self.dim_avars[name] = list(np.arange(start,stop,step))
        if not np.isclose(self.dim_avars[name][-1],stop):
            self.dim_avars[name].append(stop)
        self.dim_avars[name] = [float(v) for v in self.dim_avars[name]]

    # Set a dimension by logspace
    def set_dimension_logspace(self,name:str,start:float,stop:float,count:int):
        self.dim_avars[name] = list(np.logspace( np.log10(start) , np.log10(stop) , count))
        self.dim_avars[name] = [float(v) for v in self.dim_avars[name]]

    # Set a dimension directly
    def set_dimension_direct(self,name:str,values:list,sort:bool=True):
        the_list = list(set(values))  # remove duplicates
        if (not isinstance(values[0],str)) and sort: # sort if numeric
            the_list = sorted(the_list)
        self.dim_avars[name] = the_list

    # Print current setup
    def print_setup(self):

        log.info("Grid configuration")
        log.info(" -- name     : %s" % self.name)
        log.info(" ")
        for name in self.dim_names:
            idx = self._get_idx(name)

            log.info(" -- dimension: %s" % name)
            log.info("    parameter: %s" % self.dim_param[idx])
            log.info("    values   : %s" % self.dim_avars[name])
            log.info("    length   : %d" % len(self.dim_avars[name]))
            log.info(" ")

    # Print generated grid
    def print_grid(self):
        log.info("Flattened grid points")
        for i,gp in enumerate(self.flat):
            log.info("    %d: %s" % (i,gp))
        log.info(" ")

    def _get_tmpcfg(self, idx:int):
        return os.path.join(self.cfgdir, str(self.CONFIG_BASENAME % idx)+".toml")

    # Generate the Grid based on the current configuration
    def generate(self):
        log.info("Generating parameter grid")

        # Validate
        if len(self.dim_avars) == 0:
            raise Exception("Grid is empty")

        # Dimension count
        log.info("    %d dimensions" % len(self.dim_names))

        # Calculate total Grid size
        values = list(self.dim_avars.values())
        self.size = 1
        for v in values:
            self.size *= len(v)
        log.info("    %d points expected" % self.size)

        # Create flattened parameter grid
        flat_values = list(itertools.product(*values))
        log.info("    created %d grid points" % len(flat_values))

        # Re-assign keys to values
        log.info("    mapping keys")
        for fl in flat_values:
            gp = {}

            for i in range(len(fl)):
                gp[self.dim_param[i]] = fl[i]

            self.flat.append(gp)

        self.size = len(self.flat)
        log.info("    done")
        log.info(" ")


    def write_config_files(self):
        """Write config files."""
        # Read base config file
        base_config = read_config_object(self.conf)

        # Loop over grid points to write config files
        log.info("Writing config files")
        for i,gp in enumerate(self.flat):

            # Create new config
            thisconf:Config = deepcopy(base_config)

            # Set case dir relative to PROTEUS/output/
            thisconf.params.out.path = self.name+"/" + str(self.CONFIG_BASENAME%i)+"/"

            # Set other parameters in Config object
            for key in gp.keys():
                recursive_setattr(thisconf, key, gp[key])

            # Write this configuration file
            thisconf.write(self._get_tmpcfg(i))
            os.sync()


    def run(self,num_threads:int,
            test_run:bool=False,
            check_interval:float=15.0,
            print_interval:float=8):
        '''
        Run GridPROTEUS on the current machine.

        The value of `num_threads` will be restricted to the number of available CPUs.

        Parameters
        ----------
        - `num_threads:int`         number of CPU cores to use
        - `test_run:bool`           if true, does not actually run PROTEUS
        - `check_interval:float`    interval [secs] at which to check status of workers
        - `print_interval:int`      step interval at which to print (8*15 seconds = 2 minutes)
        '''

        log.info("Running PROTEUS across parameter grid '%s'" % self.name)

        time_start = datetime.now()
        log.info("Output path: '%s'" % self.outdir)
        if self.using_symlink:
            log.info("Symlink target: '%s'" % self.symlink_dir)
        # do not need more threads than there are points
        num_threads = min(num_threads, self.size)

        # do not use more threads than are available
        num_threads = min(num_threads, os.cpu_count())

        # Print warning
        if not test_run:
            log.info(" ")
            log.info("Confirm that the grid above is what you had intended.")
            log.info("There are %d grid points. There will be %d threads." % (self.size, num_threads))
            log.info("Do not close this program! It must stay alive to manage each of the subproceses.")
            log.info(" ")

            log.info("Sleeping...")
            for i in range(7,0,-1):
                log.info("    %d " % i)
                time.sleep(1.0)
            log.info(" ")

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
            waitfor   = 4.0
            for _ in range(int(waitfor*100)):  # do n=100*wait_for checks
                time.sleep(0.01)
                if os.path.exists(cfg_path):
                    cfgexists = True
                    break
            if not cfgexists:
               raise Exception("Config file could not be found for case %d!" % i)

            # Add process
            threads.append(multiprocessing.Process(
                                target=_thread_target,
                                args=(cfg_path,test_run,check_interval*3)
                            ))

        # Track statuses
        # 0: queued
        # 1: running
        # 2: done
        status = np.zeros(self.size)
        done = False   # All done?
        log.info("Starting process manager (%d grid points, %d threads)" % (self.size,num_threads))
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
                        threads[i].join()   # make everything wait until it's completed
                        threads[i].close()  # release resources
                        gc.collect()

            # Print info
            count_que = np.count_nonzero(status == 0)
            count_run = np.count_nonzero(status == 1)
            count_end = np.count_nonzero(status == 2)
            if (step%print_interval == 0):
                log.info("%3d queued (%5.1f%%), %3d running (%5.1f%%), %3d exited (%5.1f%%)" % (
                        count_que, 100.0*count_que/self.size,
                        count_run, 100.0*count_run/self.size,
                        count_end, 100.0*count_end/self.size
                        )
                    )

            # Done?
            if np.count_nonzero(status == 2) == self.size:
                done = True
                log.info("All cases have exited")
                break

            # Start new?
            start_new = np.count_nonzero(status == 1) < num_threads
            if start_new:
                for i in range(self.size):
                    if status[i] == 0:
                        status[i] = 1
                        start_new = False
                        threads[i].start()
                        break

            # Short sleeps while doing initial dispatch
            if step < num_threads:
                time.sleep(2.0)
            else:
                time.sleep(check_interval)
            step += 1
            # / end while loop

        # Check all cases' status files
        if not test_run:
            for i in range(self.size):
                # find file
                status_path = os.path.join(self.outdir, self.CONFIG_BASENAME%i, "status")
                if not os.path.exists(status_path):
                    raise Exception("Cannot find status file at '%s'" % status_path)

                # read file
                with open(status_path,'r') as hdl:
                    lines = hdl.readlines()
                this_stat = int(lines[0])

                # if still marked as running, it must have died at some point
                if ( 0 <= this_stat <= 9 ):
                    log.warning("Case %06d has status=running but it is not alive. Setting status=died."%i)
                    with open(status_path,'w') as hdl:
                        hdl.write("25\n")
                        hdl.write("Error (died)\n")

        time_end = datetime.now()
        log.info("All processes finished at: "+str(time_end.strftime('%Y-%m-%d_%H:%M:%S')))
        log.info("Total runtime: %.1f hours "%((time_end-time_start).total_seconds()/3600.0))


    def slurm_config(self, max_jobs:int, test_run:bool=False,
                        max_days:int=1, max_mem:int=3):
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
        """

        max_days = int(max_days) # ensure integer
        max_jobs = min(max_jobs, self.size) # do not request more jobs than we need

        log.info("Generating PROTEUS slurm config for parameter grid '%s'" % self.name)
        log.info("Will run for up to %d days per job" % max_days)
        log.info("Will use up to %d GB of memory per job" % max_mem)
        log.info(" ")

        log.info("Output path: '%s'" % self.outdir)
        if self.using_symlink:
            log.info("Symlink target: '%s'" % self.symlink_dir)

        log.info(" ")
        self.write_config_files()

        if test_run:
            command = '/bin/echo Dummy output. Config file is at: '
        else:
            command = "proteus start --offline --config"

        log_file = os.path.join(self.logdir, 'proteus-%A_%a.log')

        string = f"""#!/bin/sh
#SBATCH -J proteus.grid.array
#SBATCH --export=ALL
#SBATCH --time={max_days}-00
#SBATCH --mem-per-cpu={max_mem}G
#SBATCH -i /dev/null
#SBATCH -o {log_file}
#SBATCH --array=0-{self.size-1}%{max_jobs}

i=$SLURM_ARRAY_TASK_ID

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


def grid_from_config(config_fpath:str, test_run:bool=False):
    '''Run GridPROTEUS using the parameters in a config file (xxx.grid.toml)'''

    # Load configuration from TOML file
    with open(config_fpath, "r") as file:
        config = toml.load(file)

    # Output folder name, created inside `PROTEUS/output/`
    folder = str(config["output"])

    # Set this string to have the output files created at an alternative location. The
    #   output 'folder' in `PROTEUS/output/` will then by symbolically linked to this
    #   alternative location. Useful for when data should be saved on a storage server.
    #   THIS MUST BE AN ABSOLUTE PATH
    symlink = str(config["symlink"])
    if symlink.strip().lower() in ("", "none", "false"):
        symlink = "_UNSET"
    else:
        if not os.path.isabs(symlink):
            raise RuntimeError("Path for symbolic link must be absolute.\n\tGot: "+symlink)

    # Use SLURM?
    use_slurm = bool(config["use_slurm"])

    # Execution limits
    max_jobs = int(config["max_jobs"])   # maximum number of concurrent tasks (e.g. 300)
    max_days = int(config["max_days"])   # maximum number of days to run (e.g. 1)
    max_mem  = int(config["max_mem"])   # maximum memory per CPU in GB (e.g. 3)

    # Base config file
    cfg_base = os.path.join(PROTEUS_DIR,str(config["ref_config"]))

    # Initialise grid object
    pg = Grid(folder, cfg_base, symlink_dir=symlink, grid_config=config_fpath)

    # Add dimensions to grid by looping over keys
    dim = -1
    for key in config.keys():

        # Skip those without dots, since they aren't config variables
        if "." not in key:
            continue

        # Add dimension
        dim += 1
        name = "param_%03d"%dim
        pg.add_dimension(name, key)

        # Handle each possible method for setting this dimension
        table = config[key]
        method = table["method"]
        if method == "direct":
            pg.set_dimension_direct(name, list(table["values"]))

        elif method == "linspace":
            pg.set_dimension_linspace(name, table["start"], table["stop"], table["count"])

        elif method == "logspace":
            pg.set_dimension_logspace(name, table["start"], table["stop"], table["count"])

        elif method == "arange":
            pg.set_dimension_arange(name, table["start"], table["stop"], table["step"])

        else:
            raise ValueError(f"Invalid method for setting dimension: {method}")

    # Print information
    pg.print_setup()
    pg.generate()

    # Sanity check grid size (cannot allow larger values due to format of file paths)
    if pg.size > 999999:
        raise ValueError("Number of grid points is too large")

    # Print each grid point if the number of points isn't too large
    if pg.size < 1e4:
        pg.print_grid()
    else:
        log.warning("Number of grid points exceeds 10000. Will not print them here.")

    # Run the grid
    if use_slurm:
        # Generate Slurm batch file, use `sbatch` to submit
        pg.slurm_config(max_jobs, test_run=test_run, max_days=max_days, max_mem=max_mem)
    else:
        # Alternatively, let grid_proteus.py manage the jobs
        pg.run(max_jobs, test_run=test_run, check_interval=10)
        log.info("GridPROTEUS finished")
