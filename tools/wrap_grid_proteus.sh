#!/bin/bash
#SBATCH --export=ALL
#SBATCH --time=5-00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --job-name=grid_proteus
#SBATCH --mem-per-cpu=3G
./tools/grid_proteus.py
echo "Done with sbatch on grid_proteus"
