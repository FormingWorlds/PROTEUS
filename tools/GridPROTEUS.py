#!/usr/bin/env python3

# Run PROTEUS for a pspace of parameters

# Prepare
import os, itertools, time, subprocess, shutil, glob
import numpy as np
COUPLER_DIR=os.getenv('COUPLER_DIR')

# Object for handling the parameter space
class Pspace():

    def __init__(self, name:str, base_config_path:str):

        # Pspace's own name (for versioning, etc.)
        self.name = str(name)
        self.outdir = COUPLER_DIR+"/output/pspace_"+self.name+"/"
        self.conf = str(base_config_path)
        if not os.path.exists(self.conf):
            raise Exception("Base config file '%s' does not exist!" % self.conf)
        
        # Make output dir
        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)
        os.makedirs(self.outdir)


        # List of dimension names and parameter names
        self.dim_names = []     # String
        self.dim_param = []     # Dimension parameter of interest ('_hyper' for hypervariables)

        # Dimension variables (incl hypervariables)
        self.dim_avars = {}  

        # Flattened pspace 
        self.flat = []   # List of grid points, each is a dictionary
        self.hash = []   # Hash values of each point, for removing duplicates
    
    # Add a new empty dimension to the pspace
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
    def print_setup(self):
        print("Current setup")
        print(" -- name     : %s" % self.name)
        print(" ")
        for name in self.dim_names:
            idx = self._get_idx(name)
            
            print(" -- dimension: %s" % name)
            print("    parameter: %s" % self.dim_param[idx])
            print("    values   : %s" % self.dim_avars[name])
            print(" ")

    # Print generated space
    def print_space(self):
        print("Flattened grid points")
        for i,gp in enumerate(self.flat):
            print("    %d: %s" % (i,gp))
        print(" ")
    

    # Generate the pspace based on the current configuration
    def generate(self):
        print("Generating parameter space")

        # Validate
        if len(self.dim_avars) == 0:
            raise Exception("pspace is empty")
        
        # Dimension count
        print("    %d dimensions" % len(self.dim_names))
        
        # Calculate total pspace size
        values = list(self.dim_avars.values())
        self.size = 1
        for v in values:
            self.size *= len(v)
        print("    %d points expected" % self.size)

        # Create flattened parameter space 
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


    # Run PROTEUS across this pspace
    def run(self,test_run:bool=False):
        print("Running PROTEUS across parameter space '%s'..." % self.name)

        name = self.name.strip()

        if not test_run:
            print(" ")
            print("This program will generate several screen sessions")
            print("To kill all of them, run `pkill -f pgrid_%s_`" % name)   # (from: https://stackoverflow.com/a/8987063)
            print("Take care to avoid orphaned instances by using `screen -ls`")
            print(" ")

        # Read base config file
        with open(self.conf, "r") as f:
            base_config = f.readlines()

        # Loop over grid points
        print("Dispatching cases...")
        for i,gp in enumerate(self.flat):  

            cfgfile = self.outdir+"case_%03d.cfg" % i
            logfile = self.outdir+"case_%03d.log" % i
            screen_name = "pgrid_%s_%03dx" % (name,i)

            gp["dir_output"] = "pspace_"+self.name+"/case_%03d/"%i

            # Create config file for this case
            with open(cfgfile, 'w') as hdl:
                
                hdl.write("# GridPROTEUS config file \n")
                hdl.write("# gp_index = %d \n" % i)
                hdl.write("# screen_name = %s \n" % screen_name)

                # Write lines
                for l in base_config:
                    if (l[0] == '#') or ('=' not in l): 
                        continue
                    
                    l = l.split('#')[0]
                    key = l.split('=')[0].strip()

                    # Give priority to pgrid parameters
                    if key in gp.keys():
                        hdl.write("%s = %s # this value set by grid\n" % (key,gp[key]))

                    # Otherwise, use default value
                    else:
                        hdl.write(str(l) + "\n")
            
            print("    %d%%" % ( (i+1.0)/self.size*100.0 ) ) 
                    

            # Start proteus inside a screen session
            if test_run:
                print("(test run not dispatching proteus '%s')" % screen_name)
            else:
                
                proteus_run = 'screen -S %s -L -Logfile %s -dm bash -c "python proteus.py -cfg_file %s"' % (screen_name,logfile,cfgfile)
                subprocess.run([proteus_run], shell=True, check=True)

            time.sleep(2.0)
            # End of dispatch loop

        print("    all cases running")
        time.sleep(30.0)
        for f in glob.glob(self.outdir+"/case_*.cfg"):
            os.remove(f)

    

if __name__=='__main__':
    print("Start")

    # -----
    # Define parameter space
    # -----

    ps = Pspace("test", os.getenv('COUPLER_DIR')+"/init_coupler.cfg")

    ps.add_dimension("Redox state")
    ps.set_dimension_linspace("Redox state", "fO2_shift_IW", -2, +2, 3)

    ps.add_dimension("C/H ratio")
    ps.set_dimension_logspace("C/H ratio", "CH_ratio", 0.5, 2.0, 3)

    ps.add_dimension("Planet")
    ps.set_dimension_hyper("Planet")
    ps.append_dimension_hyper("Planet", {   "#case": "TRAPPIST-1b",
                                            "mean_distance": 0.01154,  # AU, star-planet distance
                                            "mass"         : 8.21e+24, # kg, planet mass
                                            "radius"       : 7.118e+6  # m, planet surface radius
                                        })
    ps.append_dimension_hyper("Planet", {   "#case": "TRAPPIST-1f",
                                            "mean_distance": 0.03849,  # AU, star-planet distance
                                            "mass"         : 6.21e+24, # kg, planet mass
                                            "radius"       : 6.665e+6  # m, planet surface radius
                                        })
    
    # -----
    # Print state of parameter space
    # -----

    ps.print_setup()
    ps.generate()
    ps.print_space()

    # -----
    # Start PROTEUS processes
    # -----

    ps.run(test_run=False)


    # When this script ends, all cases of PROTEUS should be running
    # It does not mean that they are complete.
    
