#!/usr/bin/env -S julia

# This Julia script post-processes PROTEUS output data with a high resolution radiative
# transfer configuration (using AGNI, so you must install it first). By default it uses
# the Honeyside4096 spectral file to do this. The resultant data are stored in a a new
# file called ppr.nc, in the PROTEUS run output folder.

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
using LoggingExtras
using NCDatasets
using Glob
using Dates
using DataStructures

# Import AGNI
using AGNI
import AGNI.atmosphere as atmosphere
import AGNI.energy as energy
import AGNI.dump as dump
import AGNI.plotting as plotting
import AGNI.setpt as setpt


function update_atmos_from_nc!(atmos::atmosphere.Atmos_t, fpath::String)

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
        # @debug "$g = $(atmos.gas_vmr[g][end]*100)%"
    end

    atmosphere.calc_layer_props!(atmos)

    # Close file
    close(ds)

    return nothing
end

function postproc(output_dir::String, nsamples::Int)

    @info "Working in $output_dir"

    # use high resolution file
    # spectral_file = joinpath(ENV["FWL_DATA"], "spectral_files/Honeyside/4096/Honeyside.sf")
    # star_file = joinpath(output_dir, "data", "0.sflux")

    # use existing spectral file
    spectral_file = joinpath(output_dir, "runtime.sf")
    star_file = ""

    if !ispath(spectral_file)
        @error("Cannot find spectral file $spectral_file")
        exit(1)
    end

    # remove old ppr files
    ppr_fpath = joinpath(output_dir, "ppr.nc")
    rm(ppr_fpath, force=true)

    # read model output
    all_files = glob("*_atm.nc", joinpath(output_dir , "data"))
    nfiles = length(all_files)
    @info @sprintf("Found %d atm files \n", nfiles)

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
    if nsamples <= 2
        years = Int[all_years[1], all_years[end]]
        files = String[all_files[1], all_files[end]]
        nsamples = 2
        @info @sprintf("Sampled down to %d files \n", nsamples)
        @debug repr(years)

    elseif nsamples < nfiles
        years = Int[]
        files = String[]
        stride::Int = Int(ceil(nfiles/(nsamples-1)))
        for i in range(start=1, step=stride, stop=nfiles)
            push!(years, all_years[i])
            push!(files, all_files[i])
        end
        push!(years, all_years[end])
        push!(files, all_files[end])
        @info @sprintf("Sampled down to %d files \n", nsamples)
        @debug repr(years)

    else
        @info "Processing all files"
        years = copy(all_years)
        files = copy(all_files)
        nsamples = nfiles
    end

    # Setup initial atmos struct...
    ref_fpath = files[1]
    @info @sprintf("Setup atmos from %s \n", ref_fpath)

    # Read reference file
    ds = Dataset(ref_fpath,"r")

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

    # rscatter
    input_flag_rayleigh::Bool = true
    try
        input_flag_rayleigh = Bool(ds["flag_rayleigh"][1] == 'y')
    catch e
        @warn "Assuming rscatter = true"
    end

    # thermo funcs
    input_flag_thermo::Bool = true
    try
        input_flag_thermo = Bool(ds["thermo_funct"][1] == 'y')
    catch e
        @warn "Assuming thermo_funct = true"
    end

    # continuum absorption
    input_flag_continuum::Bool = true
    try
        input_flag_continuum = Bool(ds["flag_continuum"][1] == 'y')
    catch e
        @warn "Assuming continuum = true"
    end

    # original atmosphere model
    original_model::String = "UNKNOWN"
    attribs = keys(ds.attrib)
    if "JANUS_version" in attribs
        original_model = "JANUS"
    elseif "AGNI_version" in attribs
        original_model = "AGNI"
    end
    @debug "Original model was $original_model"

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
        @error("Failed to allocate atmosphere")
        exit(1)
    end

    # Setup matricies for output fluxes
    @debug "Allocate output matricies"
    band_u_lw = zeros(Float64, (nsamples, atmos.nbands))
    band_d_lw = copy(band_u_lw)
    band_n_lw = copy(band_u_lw)
    band_u_sw = copy(band_u_lw)
    band_d_sw = copy(band_u_lw)
    band_n_sw = copy(band_u_lw)

    # Which level are we storing fluxes at?
    lvl::Int = 1

    # Loop over netcdfs and post-process them
    @info(" ")
    @info("Performing radiative transfer calculations...")
    for i in 1:nsamples

        # progress
        @info @sprintf("%3d /%3d = %.1f%%  \t t=%.1e yr\n",
                        i, length(files), i*100.0/length(files), years[i])

        # set new composition and structure
        update_atmos_from_nc!(atmos, files[i])

        # set fluxes to zero
        energy.reset_fluxes!(atmos)

        # do radtrans with this composition
        energy.radtrans!(atmos, true)   # LW
        energy.radtrans!(atmos, false)  # SW

        # store data in matrix
        #    lw
        band_u_lw[i, :] .= atmos.band_u_lw[lvl, :]
        band_d_lw[i, :] .= atmos.band_d_lw[lvl, :]
        band_n_lw[i, :] .= atmos.band_n_lw[lvl, :]
        #    sw
        band_u_sw[i, :] .= atmos.band_u_sw[lvl, :]
        band_d_sw[i, :] .= atmos.band_d_sw[lvl, :]
        band_n_sw[i, :] .= atmos.band_n_sw[lvl, :]
    end

    # Write to netcdf file
    @info "Writing post-processed fluxes to $ppr_fpath"

    # Absorb output from these calls, because they spam the Debug logger
    @debug "ALL DEBUG SUPPRESSED"
    with_logger(MinLevelLogger(current_logger(), Logging.Info-200)) do

        ds = Dataset(ppr_fpath,"c")

        # Global attributes
        ds.attrib["description"]        = "Post-processed PROTEUS fluxes"
        ds.attrib["date"]               = Dates.format(now(), "yyyy-u-dd HH:MM:SS")
        ds.attrib["hostname"]           = gethostname()
        ds.attrib["username"]           = ENV["USER"]
        ds.attrib["AGNI_version"]       = atmos.AGNI_VERSION
        ds.attrib["SOCRATES_version"]   = atmos.SOCRATES_VERSION
        ds.attrib["original_model"]     = original_model

        plat::String = "Generic"
        if Sys.isapple()
            plat = "Darwin"
        elseif Sys.iswindows()
            plat = "Windows"
        elseif Sys.islinux()
            plat = "Linux"
        end
        ds.attrib["platform"] = plat

        #     Create dimensions
        defDim(ds, "nbands", atmos.nbands)  # Number of spectral bands
        defDim(ds, "nsamps", nsamples)      # Number of samples that were calculated

        #     Scalar quantities
        var_specfile =  defVar(ds, "specfile" ,String, ())     # Path to spectral file when read
        var_starfile =  defVar(ds, "starfile" ,String, ())     # Path to star file when read
        var_specfile[1] = atmos.spectral_file
        var_starfile[1] = atmos.star_file

        #    Create variables
        var_time = defVar(ds, "time",      Float64, ("nsamps",), attrib = OrderedDict("units" => "yr"))
        var_bmin = defVar(ds, "bandmin",   Float64, ("nbands",), attrib = OrderedDict("units" => "m"))
        var_bmax = defVar(ds, "bandmax",   Float64, ("nbands",), attrib = OrderedDict("units" => "m"))
        var_bdl =  defVar(ds, "ba_D_LW",   Float64, ("nbands","nsamps"), attrib = OrderedDict("units" => "W m-2"))
        var_bul =  defVar(ds, "ba_U_LW",   Float64, ("nbands","nsamps"), attrib = OrderedDict("units" => "W m-2"))
        var_bnl =  defVar(ds, "ba_N_LW",   Float64, ("nbands","nsamps"), attrib = OrderedDict("units" => "W m-2"))
        var_bds =  defVar(ds, "ba_D_SW",   Float64, ("nbands","nsamps"), attrib = OrderedDict("units" => "W m-2"))
        var_bus =  defVar(ds, "ba_U_SW",   Float64, ("nbands","nsamps"), attrib = OrderedDict("units" => "W m-2"))
        var_bns =  defVar(ds, "ba_N_SW",   Float64, ("nbands","nsamps"), attrib = OrderedDict("units" => "W m-2"))

        # Write years
        var_time[:] = years[:]

        # Write band edges
        var_bmin[:] = atmos.bands_min
        var_bmax[:] = atmos.bands_max

        # Write spectral fluxes
        for i in 1:nsamples
            for ba in 1:atmos.nbands
                var_bul[ba, i] = band_u_lw[i, ba]
                var_bdl[ba, i] = band_d_lw[i, ba]
                var_bnl[ba, i] = band_n_lw[i, ba]
                var_bus[ba, i] = band_u_sw[i, ba]
                var_bds[ba, i] = band_d_sw[i, ba]
                var_bns[ba, i] = band_n_sw[i, ba]
            end
        end

        close(ds)

    end # suppress output
    @debug "ALL DEBUG RESTORED"

    # Done with atmos struct
    @debug "Deallocate atmos"
    atmosphere.deallocate!(atmos)

    return nothing
end

# Main function
function main()::Int

    # validate CLI
    if length(ARGS) != 2
        @error("Invalid arguments. Must provide output path (str) and sampling count (int).")
        return 1
    end
    target_dir = abspath(ARGS[1])
    if !isdir(target_dir)
        @error("Path does not exist: $target_dir")
        return 1
    end
    if isnothing(tryparse(Int, ARGS[2]))
        @error("Nsamp is not an integer: $(ARGS[2])")
        return 1
    end
    Nsamp = parse(Int, ARGS[2])

    # run postprocessing
    postproc(target_dir, Nsamp)

    # done
    @info "Done"
    return 0
end

# run model
exit(main())
