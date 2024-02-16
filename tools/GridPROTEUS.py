#!/usr/bin/env python3

# Run PROTEUS for a grid of parameters

# Prepare
import os, itertools, time, subprocess, shutil, sys
import numpy as np
COUPLER_DIR=os.getenv('COUPLER_DIR')
if COUPLER_DIR == None:
    raise Exception("Environment is not activated or is setup incorrectly")

# Object for handling the parameter grid
class Pgrid():

    def __init__(self, name:str, base_config_path:str, symlink_dir:str="_UNSET"):

        # Pgrid's own name (for versioning, etc.)
        self.name = str(name)
        self.outdir = COUPLER_DIR+"/output/pgrid_"+self.name+"/"
        self.tmpdir = "/tmp/pgrid_"+self.name+"/"
        self.conf = str(base_config_path)
        if not os.path.exists(self.conf):
            raise Exception("Base config file '%s' does not exist!" % self.conf)
        
        if symlink_dir == "":
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
                subfolders = [ f.path.split("/")[-1] for f in os.scandir(self.outdir) if f.is_dir() ]
                if ".git" in subfolders:
                    raise Exception("Not emptying directory - it contains a Git repository!")
                shutil.rmtree(self.outdir)
        
        # Create new output location
        if (symlink_dir == "_UNSET"):
            # Not using symlink
            os.makedirs(self.outdir)
        else:
            # Will be using symlink
            symlink_dir = os.path.abspath(symlink_dir)
            if os.path.exists(symlink_dir):
                print("Removing old files at '%s'" % symlink_dir)
                subfolders = [ f.path.split("/")[-1] for f in os.scandir(symlink_dir) if f.is_dir() ]
                if ".git" in subfolders:
                    raise Exception("Not emptying directory - it contains a Git repository!")
                shutil.rmtree(symlink_dir)
            os.makedirs(symlink_dir)
            os.symlink(symlink_dir, self.outdir)

        # Make temp dir
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)
        os.makedirs(self.tmpdir)

        # List of dimension names and parameter names
        self.dim_names = []     # String
        self.dim_param = []     # Dimension parameter of interest ('_hyper' for hypervariables)

        # Dimension variables (incl hypervariables)
        self.dim_avars = {}  

        # Flattened pgrid
        self.flat = []   # List of grid points, each is a dictionary
        self.hash = []   # Hash values of each point, for removing duplicates
    
    # Add a new empty dimension to the pgrid
    def add_dimension(self,name:str):
        if name in self.dim_names:
            raise Exception("Dimension '%s' cannot be added twice" % name)
        
        print("Added new dimension '%s' " % name)
        self.dim_names.append(name)
        self.dim_param.append("_empty")
        self.dim_avars[name] = None
        

    def _get_idx(self,name:str):
        if name not in self.dim_names:
            raise Exception("Dimension '%s' is not initialised" % name)
        return int(self.dim_names.index(name))

    # Set a dimension by linspace
    def set_dimension_linspace(self,name:str,var:str,start:float,stop:float,count:int):
        idx = self._get_idx(name)
        if self.dim_param[idx] != "_empty":
            raise Exception("Dimension '%s' cannot be set twice" % name)
        self.dim_param[idx] = var
        self.dim_avars[name] = list(np.linspace(start,stop,count))

    # Set a dimension by logspace
    def set_dimension_logspace(self,name:str,var:str,start:float,stop:float,count:int):
        idx = self._get_idx(name)
        if self.dim_param[idx] != "_empty":
            raise Exception("Dimension '%s' cannot be set twice" % name)
        self.dim_param[idx] = var
        self.dim_avars[name] = list(np.logspace( np.log10(start) , np.log10(stop) , count))

    # Set a dimension directly
    def set_dimension_direct(self,name:str,var:str,values:list):
        idx = self._get_idx(name)
        if self.dim_param[idx] != "_empty":
            raise Exception("Dimension '%s' cannot be set twice" % name)
        self.dim_param[idx] = var
        self.dim_avars[name] = list(sorted(np.array(list(set(values)))))  # check for duplicates, sort, and cast

    # Set a dimension to take hypervariables
    def set_dimension_hyper(self,name:str):
        idx = self._get_idx(name)
        if self.dim_param[idx] != "_empty":
            raise Exception("Dimension '%s' cannot be set twice" % name)
        self.dim_param[idx] = "_hyper"
        self.dim_avars[name] = []

    # Add a new hypervariable to a _hyper dimension
    # This allows multiple variables to be considered as one
    # e.g. Earth and Mars are two cases of the 'planet' hypervariable, but 
    #      each contains their own sub-variables like mass, radius, etc.
    def append_dimension_hyper(self,name:str,value:dict):
        idx = self._get_idx(name)
        if self.dim_param[idx] != "_hyper":
            raise Exception("Dimension '%s'is not ready to take new values" % name)
        self.dim_avars[name].append(value)

    # Print current setup
    def print_setup(self, f=sys.stdout):
        print("Current setup", file=f)
        print(" -- name     : %s" % self.name, file=f)
        print(" ", file=f)
        for name in self.dim_names:
            idx = self._get_idx(name)
            
            print(" -- dimension: %s" % name, file=f)
            print("    parameter: %s" % self.dim_param[idx], file=f)
            print("    values   : %s" % self.dim_avars[name], file=f)
            print(" ", file=f)

    # Print generated grid
    def print_grid(self, f=sys.stdout):
        print("Flattened grid points", file=f)
        for i,gp in enumerate(self.flat):
            print("    %d: %s" % (i,gp), file=f)
        print(" ", file=f)

    # Generate the pgrid based on the current configuration
    def generate(self):
        print("Generating parameter grid")

        # Validate
        if len(self.dim_avars) == 0:
            raise Exception("pgrid is empty")
        
        # Dimension count
        print("    %d dimensions" % len(self.dim_names))
        
        # Calculate total pgrid size
        values = list(self.dim_avars.values())
        self.size = 1
        for v in values:
            self.size *= len(v)
        print("    %d points expected" % self.size)

        # Create flattened parameter grid 
        flat_values = list(itertools.product(*values))
        print("    created %d grid points" % len(flat_values))

        # Re-assign keys to values
        print("    mapping keys")
        for fl in flat_values:
            gp = {}

            for i in range(len(fl)):
                par = self.dim_param[i]
                val = fl[i]

                if par == "_hyper":
                    for key,value in val.items():
                        gp[key] = value
                else:
                    gp[par] = val

            self.flat.append(gp)
        
        print("    done")
        print(" ")


    # Run PROTEUS across this pgrid
    def run(self,test_run:bool=False,makelog:bool=False):
        print("Running PROTEUS across parameter grid '%s'..." % self.name)

        name = self.name.strip()

        if not test_run:
            print(" ")
            print("This program will generate several screen sessions")
            print("To kill all of them, run `pkill -f pgrid_%s_`" % name)   # (from: https://stackoverflow.com/a/8987063)
            print("Take care to avoid orphaned instances by using `screen -ls`")
            print(" ")

            print("Sleeping...")
            for i in range(5,0,-1):
                print("    %d " % i)
                time.sleep(1.0)
            print(" ")

        # Read base config file
        with open(self.conf, "r") as f:
            base_config = f.readlines()

        # Loop over grid points to write config files
        print("Writing config files")
        for i,gp in enumerate(self.flat):  

            cfgfile = self.tmpdir+"/case_%05d.cfg" % i
            screen_name = "pgrid_%s_%05dx" % (name,i)

            gp["dir_output"] = "pgrid_"+self.name+"/case_%05d"%i

            # Create config file for this case
            with open(cfgfile, 'w') as hdl:
                
                hdl.write("# GridPROTEUS config file \n")
                hdl.write("# gp_index = %d \n" % i)
                hdl.write("# screen_name = %s \n" % screen_name)

                # Write lines
                for l in base_config:
                    if ('=' not in l): 
                        continue
                    
                    l = l.split('#')[0]
                    key = l.split('=')[0].strip()

                    # Give priority to pgrid parameters
                    if key in gp.keys():
                        hdl.write("%s = %s # this value set by grid\n" % (key,gp[key]))

                    # Otherwise, use default value
                    else:
                        hdl.write(str(l) + "\n")

                # Ensure data is written to disk
                hdl.flush()
                os.fsync(hdl.fileno()) 

        print("Dispatching cases...")
        for i,gp in enumerate(self.flat):  

            cfgfile = self.tmpdir+"/case_%05d.cfg" % i
            screen_name = "pgrid_%s_%05dx" % (name,i)
            gp["dir_output"] = "pgrid_"+self.name+"/case_%05d"%i
            
            if self.size > 1:
                print("    %d%% (%d of %d)" % ( (i+1)/self.size*100.0, i+1, self.size ) ) 
                    
            # Start the model
            if test_run:
                print("    (test run not dispatching PROTEUS '%s')" % screen_name)

            else:
                cfgexists = False
                waitfor   = 4.0 
                for _ in range(int(waitfor*1.0e3)):
                    time.sleep(1.0e-3)
                    if os.path.exists(cfgfile):
                        cfgexists = True
                        break
                if not cfgexists:
                    print("ERROR: Config file could not be found for case %d!" % i)

                if makelog:
                    proteus_run = COUPLER_DIR+"/tools/RunPROTEUS.sh " + cfgfile + " " + screen_name + " y "
                else:
                    proteus_run = "screen -d -m -S " + screen_name + " python proteus.py --cfg_file " + cfgfile
                
                subprocess.run([proteus_run], shell=True, check=True, stdout=subprocess.DEVNULL)

        time.sleep(5.0)
        print("    all cases running")
        print(" ")


if __name__=='__main__':
    print("Start GridPROTEUS")

    # -----
    # Define parameter grid
    # -----

    cfg_base = os.getenv('COUPLER_DIR')+"/input/earth_gridtest.cfg"
    symlink  = "/network/group/aopp/planetary/RTP035_NICHOLLS_PROTEUS/outputs/pgrid_earth_gridtest_14"
    pg = Pgrid("earth_gridtest_14", cfg_base, symlink_dir=symlink)

    # pg.add_dimension("Planet")
    # pg.set_dimension_hyper("Planet")
    # pg.append_dimension_hyper("Planet", {   "#case": "TRAPPIST-1b",
    #                                         "mean_distance": 0.01154,  # AU, star-planet distance
    #                                         "mass"         : 8.21e+24, # kg, planet mass
    #                                         "radius"       : 7.118e+6  # m, planet surface radius
    #                                     })

    pg.add_dimension("Orbital separation")
    pg.set_dimension_direct("Orbital separation", "mean_distance", [0.1, 0.2, 0.4, 0.6, 1.0, 3.0])

    pg.add_dimension("Redox state")
    pg.set_dimension_direct("Redox state", "fO2_shift_IW", [-3.0, 0.0, 3.0, 6.0])

    pg.add_dimension("C/H ratio")
    pg.set_dimension_direct("C/H ratio", "CH_ratio", [0.01, 0.1, 1.0, 2.0])

    
    # -----
    # Print state of parameter grid
    # -----

    pg.print_setup()
    pg.generate()
    pg.print_grid()

    # -----
    # Write parameter grid to file
    # -----

    with open(os.path.join(pg.outdir,"grid.txt"),"w") as hdl:
        pg.print_setup(f=hdl)
        pg.print_grid(f=hdl)

    # -----
    # Start PROTEUS processes
    # -----

    pg.run(test_run=False, makelog=True)


    # When this script ends, all cases of PROTEUS should be running.
    # It does not mean that they are complete.
    print("Exit GridPROTEUS")
    
