#!/bin/bash
# Launcher for the AGNI parameter sweep R1-R7 against the CHILI Earth-SPIDER baseline.
#
# Each run is a full restart from iter 0 (no resume) using the corresponding
# earth_spider_R{N}.toml config. We stagger launches by 180 s to avoid
# Julia precompile races on a shared depot, and we use a per-run JULIA_DEPOT_PATH
# so each run has its own compiled cache. All runs go to background; logs land
# in output_files/chili_validation/sweep_logs/.
#
# Usage: bash launch_sweep.sh
#
# Wait time: each run takes ~3 h wall to reach the iter 62 failure regime.
# With 180 s stagger and 7 runs, the final run starts ~18 min after the first,
# and all 7 reach iter 62 within a ~30 min window starting at ~3 h.

set -u

REPO=/Users/timlichtenberg/git/PROTEUS
LOG_DIR=$REPO/output_files/chili_validation/sweep_logs
mkdir -p "$LOG_DIR"

# Base env (must be exported into each subshell)
export PYTHON_JULIAPKG_EXE=/Users/timlichtenberg/.julia/juliaup/julia-1.11.8+0.aarch64.apple.darwin14/bin/julia
export FWL_DATA=/Users/timlichtenberg/git/FWL_DATA
export FC_DIR=/Users/timlichtenberg/git/PROTEUS/AGNI/fastchem
export RAD_DIR=/Users/timlichtenberg/git/PROTEUS/socrates
export PATH=/Users/timlichtenberg/.julia/juliaup/julia-1.11.8+0.aarch64.apple.darwin14/bin:$PATH

cd "$REPO"

for R in R1 R2 R3 R4 R5 R6 R7; do
    cfg=tests/validation/chili/earth_spider_${R}.toml
    out=output/earth_spider_${R}
    log=$LOG_DIR/${R}.log
    pid_file=$LOG_DIR/${R}.pid

    # Clean previous attempt for this slot
    rm -rf "$out"

    # Per-run Julia depot to avoid precompile races
    depot=$HOME/.julia_chili_${R}
    mkdir -p "$depot"

    echo "[$(date '+%H:%M:%S')] launching $R (depot=$depot, log=$log)"
    (
        export JULIA_DEPOT_PATH="$depot:$HOME/.julia"
        /Users/timlichtenberg/miniforge3/envs/proteus/bin/proteus start \
            --offline -c "$cfg" >> "$log" 2>&1
    ) &
    echo $! > "$pid_file"

    # Stagger: 180 s between launches lets each get past Julia precompile
    if [ "$R" != "R7" ]; then
        sleep 180
    fi
done

echo "[$(date '+%H:%M:%S')] all 7 launched"
wait
echo "[$(date '+%H:%M:%S')] all 7 finished"
