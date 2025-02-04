#!/usr/bin/env -S julia

# This Julia script post-processes PROTEUS output data with a high resolution radiative
# transfer configuration (using AGNI, so you must install it first). By default it uses
# the Honeyside4096 spectral file to do this. The resultant data are stored in a a new
# file called ppr.nc, in the PROTEUS run output folder.

# Activate environment
# Assuming that this file is in PROTEUS/tools/
PROTEUS_DIR = dirname(dirname(abspath(@__FILE__)))
println("PROTEUS_DIR = $PROTEUS_DIR")

ROOT_DIR = abspath( PROTEUS_DIR , "AGNI/")
using Pkg
Pkg.activate(ROOT_DIR)

# Import system packages
using Printf
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
    end

    atmosphere.calc_layer_props!(atmos)

    # Close file
    close(ds)

    return nothing
end

function setup_atmos_from_nc!(output_dir::String, ncfile::String, spfile::String, stfile::String)

    @info @sprintf("Setup atmos from %s \n", ncfile)

    # Read reference file
    ds = Dataset(ncfile,"r")

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

    if !ispath(spfile)
        @error("Cannot find spectral file $spfile")
        exit(1)
    end

    # Setup atmosphere
    atmos = atmosphere.Atmos_t()
    atmosphere.setup!(atmos, ROOT_DIR, output_dir,
                            spfile,
                            input_inst, input_s0fact, input_albedo, input_zenith,
                            input_tsurf,
                            input_gravity, input_radius,
                            nlev_c, input_pl[end], input_pl[1],
                            input_vmrs_scalar, "",
                            flag_gcontinuum=input_flag_continuum,
                            flag_rayleigh=input_flag_rayleigh,
                            thermo_functions=input_flag_thermo,
                            overlap_method="ro"
                            )
    code = atmosphere.allocate!(atmos, stfile)
    if !code
        @error("Failed to allocate atmosphere")
        exit(1)
    end

    return atmos, original_model
end

function postproc(output_dir::String, nsamples::Int, spfile::String)

    @info "Working in $output_dir"

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
        s = split(split(f,"/")[end],"_")[1]
        push!(all_years, parse(Int, s))
    end

    # get sorting mask
    mask = sortperm(all_years)
    all_years = all_years[mask]
    all_files = all_files[mask]

    # negative sample number means to sample all
    if nsamples < 1
        nsamples = nfiles
    end

    # re-sample files
    if nsamples == 1
        # use last
        years = Int[all_years[end]]
        files = String[all_files[end]]
        nsamples = 1

    elseif nsamples == 2
        # use first and last
        years = Int[all_years[1], all_years[end]]
        files = String[all_files[1], all_files[end]]

    elseif nsamples < nfiles
        # sample file names linearly
        years = Int[]
        files = String[]
        stride::Int = Int(ceil(nfiles/(nsamples-1)))
        for i in range(start=1, step=stride, stop=nfiles)
            push!(years, all_years[i])
            push!(files, all_files[i])
        end
        push!(years, all_years[end])
        push!(files, all_files[end])

    else
        # sample all
        @info "Processing all files"
        years = copy(all_years)
        files = copy(all_files)
        nsamples = nfiles
    end

    @info @sprintf("Sampling %d files \n", nsamples)
    @debug repr(years)

    # get years at which stellar spectrum was updated
    star_files = glob("*.sflux", joinpath(output_dir , "data"))
    star_years = Int[]
    for f in star_files
        s = split(split(f,"/")[end],".")[1]
        push!(star_years, parse(Int, s))
    end
    sort!(star_years)

    # use high resolution file
    if isempty(spfile)
        spectral_file = joinpath(ENV["FWL_DATA"], "spectral_files", "Honeyside", "4096", "Honeyside.sf")
        star_file = joinpath(output_dir, "data", "$(star_years[1]).sflux")
        @info "Spectral file not provided. Will use $spectral_file"
    else
    # use existing spectral file
        spectral_file = spfile
        star_file = ""
        @info "Spectral file provided by user: $spectral_file"
    end

    # Setup initial atmos struct...
    atmos, original_model = setup_atmos_from_nc!(output_dir, files[1], spectral_file, star_file)

    # Setup matricies for output fluxes
    @debug "Allocate output matricies"
    contfunc  = zeros(Float64, (atmos.nlev_c, atmos.nbands))  # contribution function at most recent atmos update
    band_u_lw = zeros(Float64, (nsamples, atmos.nbands))
    band_d_lw = copy(band_u_lw)
    band_n_lw = copy(band_u_lw)
    band_u_sw = copy(band_u_lw)
    band_d_sw = copy(band_u_lw)
    band_n_sw = copy(band_u_lw)

    # Which level are we storing fluxes at?
    lvl::Int = 1

    # Years
    star_idx::Int = 1
    current_time::Int = years[1]

    # Loop over netcdfs and post-process them
    @info(" ")
    @info("Performing radiative transfer calculations...")
    for i in 1:nsamples

        # track time
        current_time = years[i]

        # progress
        @info @sprintf("%3d /%3d = %.1f%%  \t t=%.1e yr\n",
                        i, length(files), i*100.0/length(files), current_time)


        # update stellar spectrum?
        if star_idx < length(star_years)
            if current_time >= star_years[star_idx+1]
                # needs updating...
                @info "    stellar spectrum needs updating"

                # Iterate counter
                star_idx += 1

                # Set new target path
                star_file = joinpath(output_dir, "data", "$(star_years[star_idx]).sflux")
                @debug "    using $star_file"

                # deallocate old atmos struct
                atmosphere.deallocate!(atmos)

                # create new atmos struct with updated stellar spectrum
                atmos, original_model = setup_atmos_from_nc!(output_dir, files[i], spectral_file, star_file)
            end
        end

        # set new composition and structure
        update_atmos_from_nc!(atmos, files[i])

        # set fluxes to zero
        energy.reset_fluxes!(atmos)

        # do radtrans with this composition
        energy.radtrans!(atmos, true, calc_cf=true)   # LW
        energy.radtrans!(atmos, false)  # SW

        # store fluxes in matrix
        #    lw
        @. band_u_lw[i, :] = atmos.band_u_lw[lvl, :]
        @. band_d_lw[i, :] = atmos.band_d_lw[lvl, :]
        @. band_n_lw[i, :] = atmos.band_n_lw[lvl, :]
        #    sw
        @. band_u_sw[i, :] = atmos.band_u_sw[lvl, :]
        @. band_d_sw[i, :] = atmos.band_d_sw[lvl, :]
        @. band_n_sw[i, :] = atmos.band_n_sw[lvl, :]

        # store contribution function from this instance
        for ilev in 1:atmos.nlev_c
            for iband in 1:atmos.nbands
                contfunc[ilev, iband]  = atmos.contfunc_band[ilev, iband]
            end
        end
    end

    # Write to netcdf file
    @info "Writing post-processed fluxes to $ppr_fpath"

    plat::String = "Generic"
    if Sys.isapple()
        plat = "Darwin"
    elseif Sys.iswindows()
        plat = "Windows"
    elseif Sys.islinux()
        plat = "Linux"
    end

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
        ds.attrib["platform"] = plat

        #     Create dimensions
        defDim(ds, "nbands", atmos.nbands)  # Number of spectral bands
        defDim(ds, "nsamps", nsamples)      # Number of samples that were calculated
        defDim(ds, "nlev_c", atmos.nlev_c)  # Number of level centres (atmos.nlev_c)

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
        var_cfn =  defVar(ds, "contfunc",  Float64, ("nbands","nlev_c"))

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

        # write contribution function for last atmos sample
        for lc in 1:atmos.nlev_c
            for ba in 1:atmos.nbands
                var_cfn[ba, lc] = contfunc[lc, ba]
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
    if length(ARGS) < 2
        @error("Invalid arguments. Must provide output path (str) and sampling count (int).")
        return 1
    end

    # directory to post-process
    target_dir = abspath(ARGS[1])
    if !isdir(target_dir)
        @error("Path does not exist: $target_dir")
        return 1
    end

    # samples in time
    if isnothing(tryparse(Int, ARGS[2]))
        @error("Nsamp is not an integer: $(ARGS[2])")
        return 1
    end
    Nsamp = parse(Int, ARGS[2])

    # path to spectral file
    spfile::String = ""
    if length(ARGS) == 3
        spfile = String(ARGS[3])
        if !isfile(spfile)
            @error "Invalid spectral file provided: $spfile"
        end
    end

    # run postprocessing
    postproc(target_dir, Nsamp, spfile)

    # done
    @info "Done"
    return 0
end

# run model
exit(main())
