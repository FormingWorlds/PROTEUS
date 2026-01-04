# Installing PROTEUS

## Prerequisites

Before installing PROTEUS, ensure you have:

1. **Python 3.12** (via conda/miniconda or miniforge)
2. **Git** (or install via conda: `conda install git`)
3. **Julia** (from official installer, not package manager)
4. **20 GB disk space** (Conda ~9 GB, Julia ~2 GB, PROTEUS ~few GB)
5. **30 minutes** of your time

See [Local machine guide](../../setup/local-setup.md), [Kapteyn guide](../../setup/kapteyn-setup.md), [Habrok guide](../../setup/habrok-setup.md), or [Snellius guide](../../setup/snellius-setup.md) for system-specific pre-configuration steps.

## Installation Steps

1. Clone the repository:

   ```console
   git clone https://github.com/FormingWorlds/PROTEUS.git
   cd PROTEUS
   ```

2. Create a conda environment:

   ```console
   conda env create -f environment.yml
   conda activate proteus
   ```

3. Install PROTEUS in editable mode:

   ```console
   pip install -e .
   ```

4. Install all submodules:

   ```console
   proteus install-all --export-env
   ```

## After Installation

If you want to start running PROTEUS immediately (in the same shell where you installed it), set your environment variables:

```console
source ~/.bashrc
conda activate proteus
```

If you did not use `--export-env` flag during installation, you must manually set these environment variables in your shell rc file:

- `FWL_DATA` - Path to folder with input data
- `RAD_DIR` - Path to SOCRATES installation

When you log into the system later, the environment variables will be automatically set if you used `--export-env`, but you still need to run `conda activate proteus`.

## Updating PROTEUS

To update PROTEUS and its submodules:

```console
conda activate proteus
proteus update-all
```

## Developer Installation

For developers who want editable installs of submodules, follow these additional steps after cloning PROTEUS:

1. Set environment variables:

   ```console
   mkdir /your/local/path/FWL_DATA
   echo "export FWL_DATA=/your/local/path/FWL_DATA/" >> "$HOME/.bashrc"
   source "$HOME/.bashrc"
   ```

2. Clone PROTEUS:

   ```console
   git clone git@github.com:FormingWorlds/PROTEUS.git
   cd PROTEUS
   ```

3. Create Python environment:

   ```console
   conda create -n proteus python=3.12
   conda activate proteus
   ```

4. Install SOCRATES radiative transfer:

   ```console
   ./tools/get_socrates.sh
   echo "export RAD_DIR=$PWD/socrates/" >> "$HOME/.bashrc"
   source "$HOME/.bashrc"
   ```

5. Install AGNI atmosphere model:

   ```console
   git clone git@github.com:nichollsh/AGNI.git
   cd AGNI
   bash src/get_agni.sh 0
   cd ../
   ```

6. Install optional submodules (editable):

   ```console
   # MORS stellar evolution
   git clone git@github.com:FormingWorlds/MORS
   python -m pip install -e MORS/.
   
   # JANUS atmosphere
   git clone git@github.com:FormingWorlds/JANUS
   python -m pip install -e JANUS/.
   
   # CALLIOPE outgassing
   git clone git@github.com:FormingWorlds/CALLIOPE
   python -m pip install -e CALLIOPE/.
   
   # ARAGOG interior
   git clone git@github.com:FormingWorlds/aragog.git
   python -m pip install -e aragog/.
   
   # ZEPHYRUS escape
   git clone git@github.com:FormingWorlds/ZEPHYRUS
   python -m pip install -e ZEPHYRUS/.
   ```

7. Install PETSc (requires Python <= 3.12):

   ```console
   ./tools/get_petsc.sh
   ```

8. Install SPIDER interior model:

   ```console
   ./tools/get_spider.sh
   ```

9. Install PROTEUS framework:

   ```console
   python -m pip install -e .
   ```

10. Enable pre-commit hooks:

    ```console
    pre-commit install -f
    ```

11. Done! 🚀 Any remaining dependencies will be downloaded on first run.
