#!/usr/bin/env -S julia

# script to perform radiative-convective equilibrium postprocessing
# at multiple zenith angles based on a single PROTEUS atmosphere output file
# similar to postprocess.jl but loops over multiple angles and saves
# separate .nc files for each angle.

# Usage: julia multiprofile_postprocess.jl <output_dir> <angles>
# where <output_dir> is the PROTEUS simulation output directory
# and <angles> is a comma-separated list of zenith angles in degrees
# Example: julia multiprofile_postprocess.jl ./outputdir 0,30,60,85

PROTEUS_DIR = dirname(dirname(abspath(@__FILE__)))
println("PROTEUS_DIR = $PROTEUS_DIR")

#absolute path to AGNI environment for Julia
ROOT_DIR = abspath(PROTEUS_DIR, "AGNI/")
using Pkg
Pkg.activate(ROOT_DIR)

#import required packages
using Printf, LoggingExtras, NCDatasets, Glob, Dates, DataStructures
using AGNI
import AGNI.atmosphere as atmosphere
import AGNI.energy as energy
import AGNI.solver as solver

# function largely based on setup_atmosphere! in atmosphere.jl and setup_atmos_from_nc! from postprocess.jl but
# for different zenith angles

function setup_atmos_multiangle!(output_dir::String, ncfile::String, spfile::String, zenith_angle::Float64)
    @info @sprintf("Setting up atmosphere with custom zenith_angle=%.1f\n", zenith_angle)

    ds = Dataset(ncfile,"r")

    try
        nlev_c::Int = length(ds["p"][:])
        input_pl::Array{Float64,1} = ds["pl"][:]

        # determine min/max pressure and convert to bar
        p_top_pa = minimum(input_pl)
        p_boa_pa = maximum(input_pl)
        p_top_bar = p_top_pa / 1.0e5
        p_boa_bar = p_boa_pa / 1.0e5

        # read in gases and vmrs
        raw_gases::Array{Char,2} = ds["gases"][:,:]
        num_gas::Int = size(raw_gases)[2]
        input_gases::Array{String,1} = []
        for i in 1:num_gas
            push!(input_gases, strip(String(raw_gases[:,i])))
        end

        raw_vmrs::Array{Float64, 2} = ds["x_gas"][:,:]
        input_vmrs_scalar::Dict{String, Float64} = Dict()
        for i in 1:num_gas
            g = input_gases[i]
            input_vmrs_scalar[g] = raw_vmrs[i, end]
        end

        #get other inputs for atmosphere setup from .nc file
        input_tsurf::Float64   = ds["tmp_surf"][1]
        input_radius::Float64  = ds["planet_radius"][1]
        input_gravity::Float64 = ds["surf_gravity"][1]
        input_inst::Float64   = ds["instellation"][1]
        input_s0fact::Float64 = 1.0
        #input_s0fact::Float64 = ds["inst_factor"][1]
        input_albedo::Float64 = ds["bond_albedo"][1]

        # flags required for atmosphere setup
        input_flag_rayleigh::Bool = true
        try; input_flag_rayleigh = Bool(ds["flag_rayleigh"][1] == 'y'); catch; end
        input_flag_thermo::Bool = true
        try; input_flag_thermo = Bool(ds["thermo_funct"][1] == 'y'); catch; end
        input_flag_continuum::Bool = true
        try; input_flag_continuum = Bool(ds["flag_continuum"][1] == 'y'); catch; end

        close(ds)

        #check for user in case the spectral file is missing
        if !ispath(spfile)
            @error("Cannot find spectral file $spfile")
            exit(1)
        end

        # SETUP ATMOSPHERE with function from atmosphere.jl
        atmos = atmosphere.Atmos_t()
        atmosphere.setup!(atmos, ROOT_DIR, output_dir,
                                spfile,
                                input_inst, input_s0fact, input_albedo, zenith_angle,
                                input_tsurf,
                                input_gravity, input_radius,
                                nlev_c, p_boa_bar, p_top_bar,
                                input_vmrs_scalar, "",
                                flag_gcontinuum=input_flag_continuum,
                                flag_rayleigh=input_flag_rayleigh,
                                thermo_functions=input_flag_thermo,
                                overlap_method="ee"
                                )

        code = atmosphere.allocate!(atmos, "")
        if !code
            @error("Failed to allocate atmosphere")
            exit(1)
        end

        # Load initial state (final T-P profile form simulation) from file so we can start RCE from there
        # to ensure easier convergence than starting from scratch

        ds = Dataset(ncfile, "r")
        try
            atmos.tmp[:] .= ds["tmp"][:]
            atmos.tmp_surf = ds["tmp_surf"][1]
            atmos.tmp_magma = ds["tmagma"][1]
            atmos.tmpl[:] .= ds["tmpl"][:]
            atmos.pl[:] .= ds["pl"][:]
            atmos.p[:] .= ds["p"][:]
            atmos.p_boa = atmos.pl[end]

            raw_gases = ds["gases"][:,:]
            raw_vmrs = ds["x_gas"][:,:]
            for i in 1:size(raw_gases)[2]
                gas_name = strip(String(raw_gases[:,i]))
                atmos.gas_vmr[gas_name][:] .= raw_vmrs[i, :]
            end

            atmosphere.calc_layer_props!(atmos)
        finally
            close(ds)
        end

        return atmos
    catch e
        close(ds)
        rethrow(e)
    end
end

# Perform postprocessing at a single zenith angle. This function runs the RCE solver and saves output to a new .nc file. A different
#.nc file is created for each zenith angle because I was running into issues trying to store multiple angles in a single file or merging files.
function postprocess_at_angle(output_dir::String, atmfile::String, spfile::String, angle::Float64)
    println("\n" * "="^60)
    println("Processing zenith angle = $angle")
    println("="^60)

    # calls the SETUP ATMOSPHERE function
    atmos = setup_atmos_multiangle!(output_dir, atmfile, spfile, angle)

    # SOLVE FOR RADIATIVE-CONVECTIVE EQUILIBRIUM for the given atmosphere and zenith angle
    println("  Solving for radiative-convective equilibrium...")

    success = solver.solve_energy!(atmos,
                                    sol_type=3,
                                    chem=false,
                                    convect=true,
                                    sens_heat=true,
                                    conduct=true,
                                    latent=false,
                                    rainout=false,
                                    grey_start=false,
                                    max_steps=100,
                                    max_runtime=7200.0,
                                    modprint=1,
                                    dx_max=35.0,
                                    modplot=0,
                                    conv_atol=0.05,
                                    conv_rtol=0.01,
                                    ls_method=2,
                                    method=1,
                                    perturb_all=true
                                    )

    if !success
        @warn("Failed to converge for zenith angle $angle - using best solution found")
    else
        println("  --> Equilibrium found!")
    end

    # printout some results, similar to what is given in a PROTEUS simulation
    println("\n  RESULTS:")
    println("    Zenith angle: $angle")
    println("    LW flux at TOA: $(atmos.flux_u_lw[1]) W/m2")
    println("    SW flux down at TOA: $(atmos.flux_d_sw[1]) W/m2")
    println("    SW net at surface: $(atmos.flux_d_sw[end] - atmos.flux_u_sw[end]) W/m2")
    println("    Temperature at surface: $(atmos.tmp[end]) K")
    println("    Temperature at TOA: $(atmos.tmp[1]) K")

    # Extract and save
    temp_profile = copy(atmos.tmp[:])
    press_profile = copy(atmos.p[:])
    press_level = copy(atmos.pl[:])
    gas_names = copy(atmos.gas_names)
    nlev_c = length(atmos.tmp)
    nlev_l = length(atmos.pl)
    num_gases = length(atmos.gas_names)

    #save gases accordingly
    gas_array = zeros(Float64, num_gases, nlev_c)
    for (i, gas_name) in enumerate(gas_names)
        gas_array[i, :] = atmos.gas_vmr[gas_name][:]
    end

    #resulting fluxes if they can be helpful for analysis
    flux_u_lw = copy(atmos.flux_u_lw[:])
    flux_d_lw = copy(atmos.flux_d_lw[:])
    flux_u_sw = copy(atmos.flux_u_sw[:])
    flux_d_sw = copy(atmos.flux_d_sw[:])

    #deallocate atmosphere
    atmosphere.deallocate!(atmos)
    GC.gc(); GC.gc()
    sleep(2.0)

    # SAVE TO NETCDF with filename based on zenith angle and original atmfile
    output_file = replace(atmfile, ".nc" => "_z$(Int(round(angle))).nc")
    println("  Saving to: $(basename(output_file))")

    ds_out = Dataset(output_file, "c")
    try
        ds_out.dim["nlev_c"] = nlev_c
        ds_out.dim["nlev_l"] = nlev_l
        ds_out.dim["num_gases"] = num_gases

        v_temp = defVar(ds_out, "temperature", Float64, ("nlev_c",),
                       attrib = OrderedDict("units" => "K"))
        v_temp[:] = temp_profile

        v_press = defVar(ds_out, "pressure", Float64, ("nlev_c",),
                        attrib = OrderedDict("units" => "Pa"))
        v_press[:] = press_profile

        v_pressl = defVar(ds_out, "pressure_level", Float64, ("nlev_l",),
                         attrib = OrderedDict("units" => "Pa"))
        v_pressl[:] = press_level

        v_gas = defVar(ds_out, "gas_vmr", Float64, ("num_gases", "nlev_c"))
        v_gas[:, :] = gas_array

        v_flu_lw = defVar(ds_out, "flux_u_lw", Float64, ("nlev_l",),
                         attrib = OrderedDict("units" => "W/m2"))
        v_flu_lw[:] = flux_u_lw

        v_fld_lw = defVar(ds_out, "flux_d_lw", Float64, ("nlev_l",),
                         attrib = OrderedDict("units" => "W/m2"))
        v_fld_lw[:] = flux_d_lw

        v_flu_sw = defVar(ds_out, "flux_u_sw", Float64, ("nlev_l",),
                         attrib = OrderedDict("units" => "W/m2"))
        v_flu_sw[:] = flux_u_sw

        v_fld_sw = defVar(ds_out, "flux_d_sw", Float64, ("nlev_l",),
                         attrib = OrderedDict("units" => "W/m2"))
        v_fld_sw[:] = flux_d_sw

        ds_out.attrib["zenith_angle"] = angle
        ds_out.attrib["gas_names"] = join(gas_names, ",")
        ds_out.attrib["description"] = "Radiative-convective equilibrium at zenith angle $angle"
        ds_out.attrib["date"] = Dates.format(now(), "yyyy-u-dd HH:MM:SS")

    finally
        close(ds_out)
    end

    println("  --> COMPLETE\n")
end

# Main function to postprocess multiple angles. loops over given angles and calls postprocess_at_angle for each one
# angles can be given as a vector of Float64 values for simplicity

function postprocess_multiple_angles(output_dir::String, spfile::String, angles::Vector{Float64})
    output_dir = abspath(output_dir)
    data_dir = joinpath(output_dir, "data")

    # Find the latest atmosphere file in the data directory by looking for larger number in filenames
    files = Glob.glob("*_atm.nc", data_dir)
    nums = [parse(Int, match(r"(\d+)_atm\.nc", basename(f)).captures[1]) for f in files]
    atmfile = files[argmax(nums)]

    # Extra information for logging in case something goes wrong
    println("\n" * "="^60)
    println("MULTI-ANGLE POSTPROCESSING WITH RCE SOLVER")
    println("="^60)
    println("Atmosphere file: $atmfile")
    println("Spectral file:   $spfile")
    println("Angles: $angles")
    println("="^60)

    # calls postprocess_at_angle for each angle given by command
    for (i, angle) in enumerate(angles)
        println("\n>>> Angle $i of $(length(angles))")
        postprocess_at_angle(output_dir, atmfile, spfile, angle)
    end

    println("\n" * "="^60)
    println("*** ALL ANGLES COMPLETE ***")
    println("="^60)

    # print resulting filenames for user
    # the files are conveniently stored in the same directory as the original atmosphere file (i.e. data folder in output directory)
    for angle in angles
        println("  $(basename(replace(atmfile, ".nc" => "_z$(Int(round(angle))).nc")))")
    end
    println("="^60)
end

# Main entry point when called from command line
function main()
    if length(ARGS) < 2
        println("Usage: julia multiprofile_postprocess.jl <output_dir> <angles>")
        exit(1)
    end

    #create spectral file path, simply inside output_dir
    spfile = joinpath(abspath(ARGS[1]), "runtime.sf")

    #trigger multi-angle postprocessing with given arguments
    postprocess_multiple_angles(abspath(ARGS[1]), spfile,
                                parse.(Float64, split(ARGS[2], ",")))
end

main()
