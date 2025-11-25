# Habrok Cluster Guide

## Access the Habrok Cluster

You will need a RUG account, with an account name (e.g. `p321401`) and two-factor authentication set up.
To do this, first follow [the instructions](https://wiki.hpc.rug.nl/habrok/connecting_to_the_system/connecting) on the online documentation.

We recommend that you also add your public ssh key to Habrok. Doing so allows password-free connectivity:
```console
ssh-keygen -t rsa
ssh-copy-id -i ~/.ssh/id_rsa.pub YOUR_USERNAME@login1.hb.hpc.rug.nl
```

Once you have added your SSH key to Habrok, modify the entry below and insert it into your `~/.ssh/config` file
```
Host habrok1
    HostName interactive1.hb.hpc.rug.nl
    User YOUR_USERNAME
    IdentityFile ~/.ssh/id_rsa
    ServerAliveInterval 120
    ServerAliveCountMax 60
```

## Configure environment

Once you are connected to one of the interactive servers, use these steps to configure your environment **before running PROTEUS**.

1. We need to configure the correct modules. Run the following commands to set your bashrc file:
    ```console
    echo "module purge" >>  "$HOME/.bashrc"
    ```
    ```console
    echo "module load netCDF-Fortran libarchive"  >>  "$HOME/.bashrc"
    ```

2. Log out of Habrok, and then login again

3. You can now follow the usual installation steps [here](installation.html).

## File system partitioning

The standard size of your home folder is 50 GB. This is sufficient for installing PROTEUS and its submodules, but not for storing output data.
You should store output data in your personal folder within `/scratch/`, which is accessible to the compute nodes.
Since `/scratch/` is frequently emptied, you should then copy important data to `/projects/` for long term storage.

See the [information on the HPC wiki](https://wiki.hpc.rug.nl/habrok/job_management/partitions) for details.

The best way to organise this is to create a symbolic link from the PROTEUS output folder to `/scratch`.
Once you have installed PROTEUS, this can be done by running the following commands inside your PROTEUS folder:
```console
rm -rf output
mkdir /scratch/$USER/proteus_output
ln -sf /scratch/$USER/proteus_output output
```
Anything written to `output/` will then be stored inside the `/scratch` partition.

## Resource limits

Each PROTEUS simulation should be allocated at least 2 GB of memory - ideally 3 GB.
Habrok [limits job runtime](https://wiki.hpc.rug.nl/habrok/job_management/partitions) to a maximum of 10 days on the "regular" node partition.
The parallel and GPU partitions are limited to 5 and 3 days respectively; they should be avoided since PROTEUS will not benefit from these.

## Submitting jobs, and running grids of simulations

There is information on the HPC wiki on [how to submit jobs](https://wiki.hpc.rug.nl/habrok/job_management/scheduling_system) with SLURM.

- To submit a generic script, run:
    ```console
    sbatch name_of_your_script.sh
    ```

- To check the status of your jobs, use:
    ```console
    squeue -u $USER
    ```

See the section on running grids in the PROTEUS [usage guide](./usage.html#running-grids-of-simulations). These instructions will detail how to submit grids to the nodes via SLURM.

You can also submit single PROTEUS runs to the nodes. For example:
```console
sbatch --mem-per-cpu=3G --time=1440 --wrap "proteus start -oc input/all_options.toml"
```
