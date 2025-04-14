# Habrok Cluster Guide

## Access the Habrok Cluster

You will need a RUG account, with an account name  (e.g. `p321401`) and two-factor
authentication (2FA) already set up. The cluster is accessed via SSH.
Follow [the instructions](https://wiki.hpc.rug.nl/habrok/connecting_to_the_system/connecting) on the online documentation.

We recommend that you also add your public ssh key to Habrok. Doing so allows password-free
connectivity: [see instructions here](https://wiki.hpc.rug.nl/habrok/connecting_to_the_system/ssh_key_login).

## Configure environment

Once you are connected to one of the interactive servers, use these steps to configure your environment **before running PROTEUS**.

1. Add the appropriate modules to your shell RC file:
    ```console
    echo "module purge" >>  "$HOME/.bashrc"
    echo "module load netCDF-Fortran"  >>  "$HOME/.bashrc"
    ```

2. Log out of Habrok, and then login again

3. You can now follow the usual installation steps [here](./installation.md).

## File system partitioning

The standard size of your home folder is 50 GB. This is sufficient for installing PROTEUS and its submodules, but not for storing output data.
You should store output data in your personal folder within `/scratch/`, which is accessible to the compute nodes.
Since `/scratch/` is frequently emptied, you should then copy important data to `/projects/` for long term storage.

See the [information on the HPC wiki](https://wiki.hpc.rug.nl/habrok/job_management/partitions) for details.

## Submitting and Monitoring Jobs

There is information on the HPC wiki on [how to submit jobs](https://wiki.hpc.rug.nl/habrok/job_management/scheduling_system) to the Habrok cluster.

- To submit your script, run:
    ```console
    sbatch name_of_your_script.sh
    ```

- To check the status of your jobs, use:
    ```console
    squeue -u $USER
    ```

