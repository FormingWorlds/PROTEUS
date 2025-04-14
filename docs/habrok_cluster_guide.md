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

2. Use [miniforge](https://github.com/conda-forge/miniforge) to install Python 3.12
    ```console
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    ```

3. Log out of Habrok, and then login again

4. You can now follow the usual installation steps [here](./installation.md).

## Running PROTEUS on the interactive servers

You should first run PROTEUS on the interactive servers. This ensures that the model is
configured and working correctly, before using the compute nodes.

- To launch a simulation on Habrok, use the terminal command `tmux`. You can find detailed documentation [here](https://tmuxcheatsheet.com/).
- For example, you can start a new tmux session with the command:
    ```console
    tmux new -s <session_name>
    ```
- Inside the tmux session, start your first simulation:
    ```console
    proteus start --config input/all_options.toml
    ```
- To detach from the session, press `Ctrl + b`, then `d`. You can reattach to the session later with:
    ```console
    tmux attach -t <session_name>
    ```
- To list all tmux sessions, use:
    ```console
    tmux ls
    ```
- To kill a tmux session, use:
    ```console
    tmux kill-session -t <session_name>
    ```
- The above started simulation will store the output data in the PROTEUS `output/` folder. You can check the progress of the simulation by looking at the log files in this folder. The log files are named according to the simulation name and contain information about the simulation's progress and any errors that may have occurred.
- If you want to check if you are using CPUs on the cluster, use the command:
    ```console
    htop
    ```
- Press `Ctrl + c` to exit the `htop` command.


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
