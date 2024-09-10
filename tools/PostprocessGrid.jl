#!/usr/bin/env -S julia

# Activate environment
if !haskey(ENV, "PROTEUS_DIR")
    error("The PROTEUS_DIR environment variable has not been set")
end
ROOT_DIR = abspath( ENV["PROTEUS_DIR"] , "AGNI/")
using Pkg
Pkg.activate(ROOT_DIR)

using Glob
using LoggingExtras


# Wrapper
function once(dir::String, Nsamp::Int)
    execpath::String = joinpath(ENV["PROTEUS_DIR"], "tools", "Postprocess.jl")

    @info "Start processing $dir..."
    logfile = joinpath(dir, "ppr.log")
    run(pipeline(`$execpath $dir $Nsamp`, stdout=logfile, stderr=logfile))

    return nothing
end

# Main function
function main()::Int

    # validate CLI
    if length(ARGS) != 2
        @error("Invalid arguments. Most provide output path (str) and sampling count (int).")
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
        @error "No case dirs found in $target_dir"
        return 1
    end

    # run postprocessing in parallel
    @sync begin
        for d in case_dirs
            @async once(d, Nsamp)
        end
    end

    # done
    @info "Done"
    return 0
end

# run model
exit(main())
