# Activate environment
ROOT_DIR = abspath(ENV["PROTEUS_DIR"], "AGNI/")
using Pkg
Pkg.activate(ROOT_DIR)

# Set SOCRATES
ENV["RAD_DIR"] = joinpath(ROOT_DIR,"socrates")

# Import system packages
using Printf
using Plots
using LaTeXStrings
using NCDatasets

# Import AGNI
using AGNI
import AGNI.atmosphere as atmosphere
import AGNI.energy as energy
import AGNI.dump as dump
import AGNI.plotting as plotting
import AGNI.setpt as setpt

# Set simulation output folder
output_dir = joinpath(ENV["PROTEUS_DIR"], "output", )

# use same spectral file as simulation
spectral_file = joinpath(output_dir, "runtime.sf")
star_file = ""

# use high resolution file
# spectral_file = joinpath(ROOT_DIR, "res/spectral_files/nogit/Honeyside/4096/Honeyside.sf")
# star_file = joinpath(ROOT_DIR, "res/stellar_spectra/hd97658.txt")
