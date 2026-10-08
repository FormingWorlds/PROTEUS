# Cambridge IoA Cluster Guide

!!! note Limited space in your home folder
    Your home directory is very limited on storage space (just a few GB). This means that files should be stored on the data drives. Ask IT for space on these.

## Setup Visual Studio Code server

The VSCode server takes up a lot of space.

1. Disconnect any vscode connections, and re-connect using a normal terminal `ssh` session.
2. Remove the folder `rm -rf ~/.vscode-server`
3. Create it on the data drive `mkdir /data/$USER/.vscode-server`
4. Create a symbolic link to this new folder `ln -sf /data/$USER/.vscode-server ~/.vscode-server`



## Setup Julia

If you have already installed Julia in your home folder, remove it and any bashrc entries.

Then set the variables in your bashrc file:
```bash
export JULIAUP_HOME=/data/$USER/.juliaup/
export JULIAUP_DEPOT_PATH=/data/$USER/.juliaup/
export JULIA_DEPOT_PATH=/data/$USER/.julia/
```

Set your module config in your bashrc too:
```bash
module load netcdf openmpi mpich cuda
```

Then install Julia, making sure to set the path to `/data/<username>/.julia` when prompted

```bash
curl -fsSL https://install.julialang.org | sh
```

## Setup Python

It is **important that you DO NOT** set the variable `PYTHON_JULIAPKG_EXE`. The main install steps will instruct you to do this. Please ignore that advice.

Then install Python, also making sure to install to `/data/<username>/miniforge3`:

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

!!! note Special steps completed
    Return to the main installation guide.
