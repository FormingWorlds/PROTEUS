#!/bin/bash
#SBATCH --job-name=energy_test
#SBATCH --partition=regular
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-2
#SBATCH --output=/scratch/p311056/proteus_runs/logs/%x-%a-%A.log
#SBATCH --error=/scratch/p311056/proteus_runs/logs/%x-%a-%A.err

# Required modules for SOCRATES (libnetcdff, libgfortran)
module load GCC/12.3.0 netCDF-Fortran/4.6.1-gompi-2023a

# Initialize conda (NOT source ~/.bashrc)
eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
conda activate proteus

# Pin Julia 1.11 (AGNI compatibility)
export PYTHON_JULIAPKG_EXE=$HOME/.julia/juliaup/julia-1.11.9+0.x64.linux.gnu/bin/julia

cd $HOME/PROTEUS

# Map array task to config
case $SLURM_ARRAY_TASK_ID in
    1) CONFIG="tests/validation/energy_test_spider.toml" ;;
    2) CONFIG="tests/validation/energy_test_aragog.toml" ;;
esac

echo "=== Task $SLURM_ARRAY_TASK_ID: $CONFIG ==="
echo "=== $(date) ==="

proteus start --offline -c "$CONFIG"

echo "=== Done: $(date) ==="
