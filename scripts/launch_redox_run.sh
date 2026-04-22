#!/bin/zsh
# Safe launcher for PROTEUS runs against the redox scaffolding.
#
# Usage:
#   scripts/launch_redox_run.sh <config.toml> <log_path>
#
# Example:
#   scripts/launch_redox_run.sh \
#     input/chili/chili_redox_fO2init_smoke_10iter.toml \
#     /tmp/launch_redox_smoke.log
#
# This wrapper:
#   1. Activates the proteus-redox conda env.
#   2. Sets FWL_DATA, RAD_DIR, Julia, FastChem, Zalmoxis paths.
#   3. Pins OPENBLAS_NUM_THREADS=1 for reproducibility (critical for
#      any A/B diff work; keeps scaffolding runs consistent).
#   4. Runs preflight_redox_env.sh to assert proteus imports from
#      PROTEUS-redox/src. Bails on mismatch.
#   5. Prints git HEAD and module resolution for the log.
#   6. Launches PROTEUS in-foreground so nohup/caller controls
#      backgrounding.
set -e

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config.toml> [log_path]" >&2
    exit 2
fi
CONFIG="$1"

cd /Users/timlichtenberg/git/PROTEUS-redox
eval "$(/Users/timlichtenberg/miniforge3/bin/conda shell.zsh hook)"
conda activate proteus-redox

export FWL_DATA=/Users/timlichtenberg/git/FWL_DATA
export RAD_DIR=/Users/timlichtenberg/git/PROTEUS/socrates/
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHON_JULIAPKG_EXE="$HOME/.julia/juliaup/julia-1.11.8+0.aarch64.apple.darwin14/bin/julia"
export FC_DIR="$HOME/git/PROTEUS/AGNI/fastchem"
export ZALMOXIS_ROOT=/Users/timlichtenberg/git/PROTEUS-redox/Zalmoxis

echo "==== REDOX launch at $(date) ===="
echo "config: $CONFIG"
echo "python: $(which python)"
echo "git HEAD: $(git rev-parse HEAD) on $(git branch --show-current)"

source scripts/preflight_redox_env.sh

proteus start --offline -c "$CONFIG"
echo "==== REDOX launch finished at $(date) ===="
