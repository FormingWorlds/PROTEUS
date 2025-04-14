#!/bin/bash
#SBATCH --export=ALL
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --job-name=grid_proteus
#SBATCH --mem=60000
./tools/grid_proteus.py
echo "Done with sbatch on grid_proteus"
