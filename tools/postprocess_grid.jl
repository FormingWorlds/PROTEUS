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
function once(dir::String, Nsamp::Int)
    execpath::String = joinpath(PROTEUS_DIR, "tools", "Postprocess.jl")

    @info "Start processing $dir..."
    logfile = joinpath(dir, "ppr.log")
    run(pipeline(`$execpath $dir $Nsamp`, stdout=logfile, stderr=logfile))

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

    # find case dirs
    case_dirs = glob("case_*", target_dir)
    if isempty(case_dirs)
        @error "No case subdirectories found"
        return 1
    end

    # threads counter
    @info "Will use up to $(Threads.nthreads()) threads"
    sleep(3)

    # run postprocessing in parallel
    tasks = Task[]
    @sync begin
        for d in case_dirs
            push!(tasks, @async once(d, Nsamp))
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
