#!/usr/bin/env python3

# Run PROTEUS for a grid of parameters

from __future__ import annotations

import itertools
import logging
import os
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from proteus.config import Config, read_config_object
from proteus.utils.helper import get_proteus_dir

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

# Object for handling the parameter grid
class Grid():

    def __init__(self, name:str, base_config_path:str, symlink_dir:str="_UNSET"):

        # Grid's own name (for versioning, etc.)
        self.name = str(name).strip()
        self.outdir = PROTEUS_DIR+"/output/"+self.name+"/"
        self.tmpdir = "/tmp/"+self.name+"/"
        self.conf = str(base_config_path)
        if not os.path.exists(self.conf):
            raise Exception("Base config file '%s' does not exist!" % self.conf)

        self.symlink_dir = symlink_dir
        if self.symlink_dir in ["","."]:
            raise Exception("Symlinked directory is set to a blank path")

        # Paths
        self.outdir = os.path.abspath(self.outdir)
        self.tmpdir = os.path.abspath(self.tmpdir)

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

        # Make temp dir
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)
        os.makedirs(self.tmpdir)

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
        return os.path.join(self.tmpdir,"case_%05d.toml" % idx)

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


    def slurm_config(self, max_jobs:int=10, test_run:bool=False):
        log.info("Running PROTEUS across parameter grid '%s'" % self.name)

        log.info("Output path: '%s'" % self.outdir)
        if self.using_symlink:
            log.info("Symlink target: '%s'" % self.symlink_dir)

        # Read base config file
        base_config = read_config_object(self.conf)

        # Loop over grid points to write config files
        log.info("Writing config files")
        for i,gp in enumerate(self.flat):

            # Create new config
            thisconf:Config = deepcopy(base_config)

            # Set case dir relative to PROTEUS/output/
            path = f"{self.name}/case_{i:05d}"
            thisconf.params.out.path = path

            # Set other parameters
            for key in gp.keys():
                val = gp[key]
                bits = key.split(".")
                depth = len(bits)-1

                # Descend down attributes tree until we are setting the right value
                #    This could be done with recursion, but we only ever go 2 layers deep
                #    at most, so it can easily be handled by 'brute force' like this.
                if depth == 0:
                    setattr(thisconf,bits[0],val)
                elif depth == 1:
                    setattr(getattr(thisconf,bits[0]), bits[1],val)
                elif depth == 2:
                    setattr(   getattr( getattr(thisconf,bits[0]),  bits[1] ), bits[2],val)
                else:
                    raise Exception("Requested key is too deep for configuration tree")

            # Write this configuration file
            thisconf.write(self._get_tmpcfg(i))
            os.sync()

        commands = []

        for i in range(self.size):
            cfg_path = self._get_tmpcfg(i)

            if not os.path.exists(cfg_path):
               raise IOError(f"Config file could not be found for case {i}!")

            if test_run:
                command = f'/bin/echo Dummmy output. Config file is at {cfg_path}'
            else:
                command = f"proteus start --offline --config {cfg_path}"

            commands.append(command)

        scripts = '\n'.join(f'    {cmd}' for cmd in commands)

        array_size = len(commands)

        logs_dir = Path(self.outdir)

        out_file = logs_dir / 'proteus-%A_%a.out'
        err_file = logs_dir / 'proteus-%A_%a.err'

        string = f"""#!/bin/sh
#SBATCH -J proteus.grid.array
#SBATCH -i /dev/null
#SBATCH -o {out_file}
#SBATCH -e {err_file}
#SBATCH --array=0-{array_size-1}%{max_jobs}

scripts=(
{scripts}
)
i=$SLURM_ARRAY_TASK_ID
while [ $i -le {len(commands)} ]; do
    echo executing ${{scripts[$i]}}
    ${{scripts[$i]}} || true
    i=$((i+{array_size}))
done
"""

        fname = 'proteus_slurm_array.sh'
        with open(fname, 'w') as f:
            f.write(string)

        log.info('Slurm file written to %s', fname)
        log.info('Submit to Slurm using `sbatch %s`', fname)


if __name__=='__main__':
    print("Start GridPROTEUS")

    # Output folder name, created inside `PROTEUS/output/`
    folder = "grid_test"

    # Base config file
    config = "demos/dummy.toml"
    cfg_base = os.path.join(PROTEUS_DIR,"input",config)

    # Set this string to have the output files created at an alternative location. The
    #   output 'folder' in `PROTEUS/output/` will then by symbolically linked to this
    #   alternative location. Useful for when data should be saved on a storage server.
    # symlink = "/network/group/aopp/planetary/RTP035_NICHOLLS_PROTEUS/outputs/"+folder
    symlink = None

    # Initialise grid object
    pg = Grid(folder, cfg_base, symlink_dir=symlink)

    # Add dimensions to grid...
    pg.add_dimension("Redox state", "outgas.fO2_shift_IW")
    pg.set_dimension_direct("Redox state", [-4.5, -4.0, -3.5, -3, -2.5, -2.0, -1.5, -1.0])

    pg.add_dimension("Hydrogen", "delivery.elements.H_ppmw")
    pg.set_dimension_direct("Hydrogen", [16000, 13000, 10000, 7000, 4000, 1000], sort=False)

    pg.add_dimension("Sulfur", "delivery.elements.SH_ratio")
    pg.set_dimension_direct("Sulfur", [2, 7, 12])

    pg.add_dimension("Mass", "struct.mass_tot")
    pg.set_dimension_direct("Mass", [1.85, 2.14, 2.39])

    pg.print_setup()
    pg.generate()
    pg.print_grid()

    pg.slurm_config(max_jobs=10, test_run=False)
