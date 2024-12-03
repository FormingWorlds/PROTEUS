#!/usr/bin/env -S julia --threads auto

# This Julia script is used to parallelise postprocessing performed by `Postprocess.jl` for
# instances where PROTEUS has been executed across a grid. All grid points will be processed
# in parallel, using one thread each. See the content of `Postprocess.jl` for more info.

# Activate environment
# Assuming that this file is in PROTEUS/tools/
PROTEUS_DIR = dirname(dirname(abspath(@__FILE__)))
println("PROTEUS_DIR = $PROTEUS_DIR")

ROOT_DIR = abspath( PROTEUS_DIR , "AGNI/")
using Pkg
Pkg.activate(ROOT_DIR)

using Glob
using LoggingExtras

# Wrapper
function once(dir::String, Nsamp::Int, spfile::String)
    execpath::String = joinpath(PROTEUS_DIR, "tools", "postprocess.jl")

    @info "Start processing $dir..."
    logfile = joinpath(dir, "ppr.log")
    run(pipeline(`$execpath $dir $Nsamp $spfile`, stdout=logfile, stderr=logfile))

    return nothing
end

# Main function
function main()::Int

    # validate CLI
    if length(ARGS) < 2
        @error("Invalid arguments. Must provide output path (str) and sampling count (int).")
        return 1
    end

    # grid base directory
    target_dir = abspath(ARGS[1])
    if !isdir(target_dir)
        @error("Path does not exist: $target_dir")
        return 1
    end

    # time samples per case
    if isnothing(tryparse(Int, ARGS[2]))
        @error("Nsamp is not an integer: $(ARGS[2])")
        return 1
    end
    Nsamp = parse(Int, ARGS[2])

    # optimise by using first case spectral file
    optimise::Bool = false
    spfile::String = ""
    if length(ARGS) >= 3
        for arg in ARGS
            optimise = optimise || (lowercase(arg) == "optimise")
        end
        if optimise
            @info "Optimising file I/O"
            @info "    will use case00000 spectral file for all cases"
        end
    end

    # find case dirs
    case_dirs = glob("case_*", target_dir)
    if isempty(case_dirs)
        @error "No case subdirectories found"
        return 1
    end

    # threads counter
    @info "Will use up to $(Threads.nthreads()) threads"
    sleep(3)

    # remove old files
    for c in case_dirs
        rm(joinpath(c, "runtime.sf")  , force=true)
        rm(joinpath(c, "runtime.sf_k"), force=true)
        rm(joinpath(c, "ppr.nc"),       force=true)
        rm(joinpath(c, "ppr.log"),      force=true)
    end

    # run first case
    if optimise
        # run
        once(case_dirs[1], Nsamp, spfile)

        # get spectral file
        spfile = joinpath(case_dirs[1],"runtime.sf")
        if !isfile(spfile)
            @error "First case failed. Could not find spectral file in output. Exiting."
            exit(1)
        end
    end

    # run postprocessing in parallel
    tasks = Task[]
    @sync begin
        for d in case_dirs
            push!(tasks, @async once(d, Nsamp, spfile))
        end
    end

    # wait for tasks to finish
    for t in tasks
        wait(t)
    end

    # done
    @info "Done"
    return 0
end

# run model
exit(main())
