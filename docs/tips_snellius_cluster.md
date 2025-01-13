# Install PROTEUS on the Snellius Cluster

1. Connect via SSH. See instructions from SURF [here](https://servicedesk.surf.nl/wiki/display/WIKI/SSH+public-key+authentication).

2. Set up your working environment. You have to load some modules and set up the environment variables prior to the installation. To facilitate this we suggest to copy the following function into your `.bashrc` file of your home directory. Then, simply run this function `boot_PROTEUS` in your terminal each time you login.

    ```console
    boot_PROTEUS () {
      module load 2023
      module load Python/3.11.3-GCCcore-12.3.0
      module load SciPy-bundle/2023.07-gfbf-2023a
      module load netCDF-Fortran/4.6.1-gompi-2023a

      export FWL_DATA=${HOME}/your_path_to_fwl_data
      export RAD_DIR=${HOME}/your_path_to_socrates
    }
    ```

3. Follow the generic installation instructions [here](./installation.md) from "Download the framework". We then recommand to add the activation of the python virtual environment into the `boot_PROTEUS` function of step 2.

    ```console
    boot_PROTEUS () {
      [...]

      source ${HOME}/proteus_directory/.venv/bin/activate
    }
    ```
