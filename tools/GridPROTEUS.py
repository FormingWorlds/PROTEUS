#!/usr/bin/env python3

# Run PROTEUS for a pspace of parameters

# Prepare
import os, json, itertools
import numpy as np
COUPLER_DIR=os.getenv('COUPLER_DIR')

# Object for handling the parameter space
class Pspace():

    def __init__(self, name:str, base_config_path:str):

        # Pspace's own name (for versioning, etc.)
        self.name = str(name)
        self.conf = str(base_config_path)
        if not os.path.exists(self.conf):
            raise Exception("Base config file '%s' does not exist!" % self.conf)

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

    # Set a dimension to manually take hypervariables
    def set_dimension_hyper(self,name:str):
        idx = self._get_idx(name)
        if self.dim_param[idx] != "_empty":
            raise Exception("Dimension '%s' cannot be set twice" % name)
        self.dim_param[idx] = "_hyper"
        self.dim_avars[name] = []

    # Add a new value (e.g. hypervariable) to a dimension of type 2
    def append_dimension_hyper(self,name:str,value:dict):
        idx = self._get_idx(name)
        if self.dim_param[idx] != "_hyper":
            raise Exception("Dimension '%s'is not ready to take new values" % name)
        self.dim_avars[name].append(value)

    # Print current setup
    def print_setup(self):
        print("Current parameter space setup")
        print("    name     :%s" % self.name)
        print(" ")
        for name in self.dim_names:
            idx = self._get_idx(name)
            
            print("    dimension: %s" % name)
            print("    parameter: %s" % self.dim_param[idx])
            print("    values   : %s" % self.dim_avars[name])
            print(" ")

    # Print generated space
    def print_space(self):
        print("Current parameter space")
        for i,gp in enumerate(self.flat):
            print("    %d: %s" % (i,gp))
    

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


    # Run PROTEUS across this pspace
    def run(self):
        print("Running PROTEUS across parameter space '%s'..." % self.name)




    

if __name__=='__main__':
    print("Start")

    # -----
    # Define parameter space
    # -----

    ps = Pspace("test", os.getenv('COUPLER_DIR')+"/init_coupler.cfg")

    ps.add_dimension("Redox state")
    ps.set_dimension_linspace("Redox state", "fO2_shift_IW", -2, +2, 3)

    ps.add_dimension("C/H ratio")
    ps.set_dimension_linspace("C/H ratio", "CH_ratio", 0.5, 2.0, 3)

    ps.add_dimension("Planet")
    ps.set_dimension_hyper("Planet")
    ps.append_dimension_hyper("Planet", {   ".case": "TRAPPIST-1b",
                                            "mean_distance": 0.01154,  # AU, star-planet distance
                                            "mass"         : 8.21e+24, # kg, planet mass
                                            "radius"       : 7.118e+6  # m, planet surface radius
                                        })
    ps.append_dimension_hyper("Planet", {   ".case": "TRAPPIST-1f",
                                            "mean_distance": 0.03849,  # AU, star-planet distance
                                            "mass"         : 6.21e+24, # kg, planet mass
                                            "radius"       : 6.665e+6  # m, planet surface radius
                                        })
    
    ps.print_setup()
    ps.generate()

    ps.print_space()



    print("Done!")

    
