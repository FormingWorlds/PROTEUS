#!/bin/zsh
# Safe launcher for PROTEUS runs against the main-branch
# tl/interior-refactor worktree.
#
# Usage:
#   scripts/launch_main_run.sh <config.toml> [extra proteus args...]
#
# Examples:
#   scripts/launch_main_run.sh output_files/chili_dry_coupled_v3f_1b_full.toml
#   scripts/launch_main_run.sh output_files/my_config.toml --resume
#
# This wrapper:
#   1. Activates the proteus conda env.
#   2. Sets FWL_DATA, RAD_DIR, Zalmoxis paths matching the production
#      main-branch convention (v4 baseline uses these).
#   3. Pins OPENBLAS_NUM_THREADS=1 and MKL_NUM_THREADS=1 for
#      reproducibility (critical for any A/B diff work, and to reduce
#      thread contention inside PROTEUS's 113-thread process).
#   4. Runs preflight_main_env.sh to assert proteus imports from
#      /Users/timlichtenberg/git/PROTEUS/src. Bails on mismatch.
#   5. Prints git HEAD and module resolution for the log.
#   6. Launches PROTEUS in-foreground so nohup/caller controls
#      backgrounding.
set -e

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config.toml> [extra proteus args...]" >&2
    exit 2
fi
CONFIG="$1"
shift

cd /Users/timlichtenberg/git/PROTEUS
eval "$(/Users/timlichtenberg/miniforge3/bin/conda shell.zsh hook)"
conda activate proteus

export FWL_DATA=/Users/timlichtenberg/work/fwl_data
export RAD_DIR=/Users/timlichtenberg/git/PROTEUS/SOCRATES
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ZALMOXIS_ROOT=/Users/timlichtenberg/git/PROTEUS/Zalmoxis

echo "==== MAIN launch at $(date) ===="
echo "config: $CONFIG"
echo "python: $(which python)"
echo "git HEAD: $(git rev-parse HEAD) on $(git rev-parse --abbrev-ref HEAD)"
echo "aragog HEAD: $(git -C aragog rev-parse HEAD) on $(git -C aragog rev-parse --abbrev-ref HEAD)"

source scripts/preflight_main_env.sh

proteus start --offline -c "$CONFIG" "$@"
echo "==== MAIN launch finished at $(date) ===="
