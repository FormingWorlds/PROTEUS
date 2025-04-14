# Habrok Cluster Guide

## Access the Habrok Cluster

You will need a RUG account, with an account name  (e.g. `p321401`) and two-factor
authentication (2FA) already set up. The cluster is accessed via SSH.
Follow [the instructions](https://wiki.hpc.rug.nl/habrok/connecting_to_the_system/connecting) on the online documentation.

We recommend that you also add your public ssh key to Habrok. Doing so allows password-free
connectivity: [see instructions here](https://wiki.hpc.rug.nl/habrok/connecting_to_the_system/ssh_key_login).

## Configure environment

Once you are connected to one of the interactive servers, use these steps to configure
your environment for running PROTEUS.

1. Add the appropriate modules to your shell RC file:
    ```console
    echo "module purge" >>  "$HOME/.bashrc"
    echo "module load netCDF-Fortran"  >>  "$HOME/.bashrc"
    ```

2. Log out of Habrok, and then login again

3. You can now follow the usual installation steps [here](./installation.md).

### Submitting and Monitoring Jobs
- To submit your script, run: ```condor_submit name_of_your_script.submit```
    ```console
    condor_submit name_of_your_script.submit
    ```

- To check the status of your job, use:
    ```console
    condor_q
    ```
or
    ```console
    condor_q -better-analyze
    ```
The second command provides a more detailed job status analysis.

- Another useful command is
    ```console
    condor_status
    ```
This displays the jobs currently running on Condormaster, including both your jobs and those of other users.

### Exiting Condormaster
- To exit Condormaster and return to Norma1/Norma2, run:
    ```console
    exit
    ```
