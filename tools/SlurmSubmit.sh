#!/bin/bash
# Submission script for GridPROTEUS
# Doesn't work very well.

# Job parameters
#SBATCH --job-name=Slurm_GridPROTEUS
#SBATCH --output=slurm_out.txt
#SBATCH --error=slurm_err.txt
#SBATCH -p priority-rp
#SBATCH --begin=now
#SBATCH --no-kill

# Resources
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 19
#SBATCH --mem-per-cpu 2000

# Operations
echo "Running slurm dispatcher"

source ~/.bashrc
conda activate proteus
module load julia 
source PROTEUS.env

srun python tools/GridPROTEUS.py 


