#!/bin/bash
# Launch the 8-run PALEOS v1.1.0 validation sweep on Mac Studio.
# 4 masses (1, 3, 5, 10 M_E) x 2 resolutions (150, 600 pts/decade), all
# parallel, all using outer_solver='newton' to mirror today's T2.1i+T2.3 baseline.
#
# Each run gets its own nohup'd background process, PID written to /tmp,
# stdout/stderr captured to output/<dir>/launch.log alongside the per-run
# proteus_00.log that PROTEUS writes natively.

REPO=/Users/timlichtenberg/git/PROTEUS
INPUT_DIR=$REPO/input/chili
OUTPUT_DIR=$REPO/output

# Environment per CLAUDE.md (today's working setup; .zshrc FWL_DATA is broken)
eval "$(/Users/timlichtenberg/miniforge3/bin/conda shell.bash hook)"
conda activate proteus
export FWL_DATA=/Users/timlichtenberg/work/fwl_data
export RAD_DIR=$REPO/socrates
export PATH=/Users/timlichtenberg/miniforge3/envs/proteus/bin:$PATH

cd "$REPO" || exit 1

CONFIGS=(
    "1me_150res"
    "1me_600res"
    "3me_150res"
    "3me_600res"
    "5me_150res"
    "5me_600res"
    "10me_150res"
    "10me_600res"
)

echo "Launching 8 PALEOS v1.1.0 validation runs at $(date)"
echo "Repo: $REPO"
echo "FWL_DATA: $FWL_DATA"
echo "RAD_DIR:  $RAD_DIR"
echo "python:   $(which python)  ($(python --version))"
echo

for label in "${CONFIGS[@]}"; do
    cfg="$INPUT_DIR/chili_paleos_v1_1_0_${label}.toml"
    out="$OUTPUT_DIR/chili_paleos_v1_1_0_${label}"
    pidfile="/tmp/paleos_v1_1_0_pid_${label}.txt"

    if [ ! -f "$cfg" ]; then
        echo "MISSING CONFIG: $cfg" >&2
        continue
    fi

    mkdir -p "$out"
    nohup proteus start --offline -c "$cfg" > "$out/launch.log" 2>&1 &
    pid=$!
    echo "$pid" > "$pidfile"
    echo "  launched ${label}: PID $pid -> $pidfile"

    sleep 5  # stagger to avoid simultaneous JAX warmup contention
done

echo
echo "All 8 runs launched. PIDs in /tmp/paleos_v1_1_0_pid_*.txt"
echo "Monitor: ps -p \$(cat /tmp/paleos_v1_1_0_pid_*.txt | tr '\n' ' ')"
