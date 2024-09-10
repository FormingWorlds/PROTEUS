#!/usr/bin/env -S julia

# Activate environment
if !haskey(ENV, "PROTEUS_DIR")
    error("The PROTEUS_DIR environment variable has not been set")
end
ROOT_DIR = abspath( ENV["PROTEUS_DIR"] , "AGNI/")
using Pkg
Pkg.activate(ROOT_DIR)

# Import system packages
using Printf
using Plots
using LaTeXStrings
using NCDatasets
using Glob

# Import AGNI
using AGNI
import AGNI.atmosphere as atmosphere
import AGNI.energy as energy
import AGNI.dump as dump
import AGNI.plotting as plotting
import AGNI.setpt as setpt


function update_atmos_from_nc!(atmos, fpath)

    ds = Dataset(fpath,"r")

    nlev_c::Int = length(ds["p"][:])

    #   gas names
    raw_gases::Array{Char,2} = ds["gases"][:,:]
    num_gas::Int = size(raw_gases)[2]
    input_gases::Array{String,1} = []
    for i in 1:num_gas
        push!(input_gases, strip(String(raw_gases[:,i])))
    end

    # gas VMRs
    raw_vmrs::Array{Float64, 2} = ds["x_gas"][:,:]
    input_vmrs::Dict{String, Array{Float64,1}} = Dict()      # dict of Arrays
    input_vmrs_scalar::Dict{String, Float64} = Dict()        # dict of Floats (surface values)
    for i in 1:num_gas
        g = input_gases[i]
        input_vmrs[g]     = zeros(Float64, nlev_c)
        input_vmrs[g][:] .= raw_vmrs[i, :]

        input_vmrs_scalar[g] = input_vmrs[g][end]
    end


    # Update the struct with new values
    atmos.instellation =  ds["instellation"][1]

    atmos.tmp_surf = ds["tmp_surf"][1]
    atmos.tmp_magma = ds["tmagma"][1]
    atmos.tmpl[:] .= ds["tmpl"][:]
    atmos.tmp[:] .=  ds["tmp"][:]

    atmos.pl[:] .= ds["pl"][:]
    atmos.p[:] .=  ds["p"][:]
    atmos.p_boa = atmos.pl[end]

    for g in input_gases
        atmos.gas_vmr[g][:] .= input_vmrs[g]
    end

    atmosphere.calc_layer_props!(atmos)

    # Close file
    close(ds);

end

function main(output_dir::String, nsamples::Int)

    # use high resolution file
    spectral_file = joinpath(ENV["FWL_DATA"], "spectral_files/Honeyside/4096/Honeyside.sf")
    star_file = joinpath(output_dir, "data", "0.sflux")

    # spectral_file = joinpath(output_dir, "runtime.sf")
    # star_file = ""

    if !ispath(spectral_file)
        error("Cannot find spectral file $spectral_file")
    end

    # read model output
    all_files = glob("*_atm.nc", joinpath(output_dir , "data"))
    nfiles = length(all_files)
    @info @sprintf("Found %d files in output folder \n", nfiles)

    # get years
    all_years = Int[]
    for f in all_files
        s = split(f,"/")
        s = split(s[end],"_")[1]
        push!(all_years, parse(Int, s))
    end

    # get sorting mask
    mask = sortperm(all_years)
    all_years = all_years[mask]
    all_files = all_files[mask]

    # re-sample files
    years = Int[]
    files = String[]
    for i in range(start=1, stop=nfiles, length=nsamples)
        push!(years, all_years[i])
        push!(files, all_files[i])
    end
    nfiles = length(files)
    @info @sprintf("Sampled down to %d files \n", nfiles)

    # Setup initial atmos struct...
    fpath = files[1]
    @info @sprintf("Setup atmos from %s \n", fpath)

    ds = Dataset(fpath,"r")

    # Get all of the information that we need
    nlev_c::Int = length(ds["p"][:])
    input_pl::Array{Float64,1} = ds["pl"][:]

    #   gas names
    raw_gases::Array{Char,2} = ds["gases"][:,:]
    num_gas::Int = size(raw_gases)[2]
    input_gases::Array{String,1} = []
    for i in 1:num_gas
        push!(input_gases, strip(String(raw_gases[:,i])))
    end

    # gas VMRs
    raw_vmrs::Array{Float64, 2} = ds["x_gas"][:,:]
    input_vmrs::Dict{String, Array{Float64,1}} = Dict()      # dict of Arrays
    input_vmrs_scalar::Dict{String, Float64} = Dict()        # dict of Floats (surface values)
    for i in 1:num_gas
        g = input_gases[i]
        input_vmrs[g]     = zeros(Float64, nlev_c)
        input_vmrs[g][:] .= raw_vmrs[i, :]

        input_vmrs_scalar[g] = input_vmrs[g][end]
    end

    # surface
    input_tsurf::Float64   = ds["tmp_surf"][1]
    input_radius::Float64  = ds["planet_radius"][1]
    input_gravity::Float64 = ds["surf_gravity"][1]

    # stellar properties
    input_inst::Float64   = ds["instellation"][1]
    input_s0fact::Float64 = ds["inst_factor"][1]
    input_albedo::Float64 = ds["bond_albedo"][1]
    input_zenith::Float64 = ds["zenith_angle"][1]

    # flags
    input_flag_rayleigh::Bool  = Bool(ds["flag_rayleigh"][1] == 'y')
    input_flag_thermo::Bool    = Bool(ds["thermo_funct"][1] == 'y')
    input_flag_continuum::Bool = Bool(ds["flag_continuum"][1] == 'y')

    # Close file
    close(ds);

    # Setup atmosphere
    atmos = atmosphere.Atmos_t()
    atmosphere.setup!(atmos, ROOT_DIR, output_dir,
                            spectral_file,
                            input_inst, input_s0fact, input_albedo, input_zenith,
                            input_tsurf,
                            input_gravity, input_radius,
                            nlev_c, input_pl[end], input_pl[1],
                            input_vmrs_scalar, "",
                            flag_gcontinuum=input_flag_continuum,
                            flag_rayleigh=input_flag_rayleigh,
                            thermo_functions=input_flag_thermo,
                            overlap_method=2
                            )
    code = atmosphere.allocate!(atmos, star_file)
    if !code
        error("Failed to allocate atmosphere")
    end

    @info(" ")
    @info("Performing radiative transfer calculations...")
    # Loop over netcdfs and post-process them
    for i in 1:length(files)

        # progress
        @info @sprintf("%3d /%3d = %.1f%% \n", i, length(files), i*100.0/length(files))

        # set new composition and structure
        fpath = files[i]
        update_atmos_from_nc!(atmos, fpath)

        # set fluxes to zero
        energy.reset_fluxes!(atmos)

        # do radtrans with this composition
        energy.radtrans!(atmos, true)   # LW
        energy.radtrans!(atmos, false)  # SW

        # write data to file
        fpath = replace(fpath, "_atm.nc" => "_ppr.nc")
        dump.write_ncdf(atmos, fpath)
    end

end

# validate CLI
if length(ARGS) != 2
    error("Invalid arguments. Most provide output path (str) and sampling count (int).")
end
output_dir = abspath(ARGS[1])
if !isdir(output_dir)
    error("Path does not exist '$output_dir'")
end
if isnothing(tryparse(Int, ARGS[2]))
    error("Invalid Nsamp; must be an integer")
end
Nsamp = parse(Int, ARGS[2])

# run model
main(output_dir, Nsamp)
