# Kapteyn Cluster Guide


## Access the Kapteyn cluster via VS Code

Follow the instructions at [VS Code Instructions Kapteyn Cluster](https://docs.google.com/document/d/1Hm1J8x9CQ10dnyDJo1iohZHU6go_hxiUR7gTD2csv-M/edit?usp=sharing) to set up your VS Code environment for the Kapteyn cluster. This allows you to use the Kapteyn cluster as a remote server, enabling you to edit PROTEUS files and run simulations directly from your local machine.

## Installation

1. If you have not followed the VS Code Instructions above, then now manage your authentication keys to avoid entering your password every time you connect (optional). You can find the instructions on the Kapteyn intranet: [How to generate authentication keys for SSH, SFTP, and SCP](https://www.astro.rug.nl/intranet/computing/index.php) (Go to Computing > Howto's > How to generate authentication keys for ssh, sftp and scp):

    ```console
    ssh-keygen -t rsa
    ```
    Press Enter to accept the default file location and enter a passphrase if desired. This will create a public/private key pair in `~/.ssh/`.
    Then, copy the public key to the Kapteyn cluster:
    ```console
    ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@kapteyn.astro.rug.nl
    ```
    You can now log in without entering your password.

2. Connect to the cluster via SSH. Use `norma2` whenever possible.

    ```console
    ssh norma2
    ```

3. Create a folder with your username in `/dataserver/users/formingworlds/`. If you cannot create a folder in there, please contact Tim Lichtenberg to get access rights.

    ```console
    mkdir -p /dataserver/users/formingworlds/<username>
    cd /dataserver/users/formingworlds/<username>
    ```

4. Follow the installation instructions [here](./installation.md). You do not need to install the `netcdf` and `netcdf-fortran` libraries, as they are already installed on the Kapteyn cluster.

5. Use [miniforge](https://github.com/conda-forge/miniforge) to install the required Python version. The recommended version is 3.12.

    ```console
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    ```

6. To avoid the cluster terminating PROTEUS jobs, increase the temporary file limit for your user by adding to your shell rc file (e.g., '~/.bashrc'):
    ```console
    echo "ulimit -Sn 4000000" >> "$HOME/.bashrc"
    echo "ulimit -Hn 5000000" >> "$HOME/.bashrc"
    ```
    Then, reload your shell rc file to make the changes effective:
    ```console
    source "$HOME/.bashrc"
    ```

## Usage of PROTEUS on the Kapteyn cluster

- To launch a simulation on the Kapteyn cluster, use the terminal command `tmux`. You can find detailed documentation [here](https://tmuxcheatsheet.com/). Avoid using `screen` sessions, as it behaves inconsistently on the cluster.
- For example, you can start a new tmux session with the command:
    ```console
    tmux new -s <session_name>
    ```
- To start a first simulation, use the command:
    ```console
    proteus start --config {PATH/TO/YOUR/PROTEUS}/input/minimal.toml
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
- The above started simulation will store the output data in the `{PROTEUS}/output/` folder. You can check the progress of the simulation by looking at the log files in the `output` folder. The log files are named according to the simulation name and contain information about the simulation's progress and any errors that may have occurred.
- If you want to check if you are using CPUs on the cluster, use the command:
    ```console
    top
    ```
- Press `Ctrl + c` to exit the `top` command.

## Queuing Manager : Condormaster

To use the queuing manager on the Kapteyn cluster, you first need to SSH into Norma1 or Norma2 (```ssh norma1```).

To access Condormaster, run the following command : ```ssh condormaster```

### Submitting a Job on Condormaster
To run a job using Condormaster, you first need to write a submit script. Begin by navigating to your home directory and creating a new submit script using : ```nano name_of_your_script.submit```

You can copy and paste the example submit script (to start a single PROTEUS simulation) below and modify it according to your needs.

```
    getenv = True
    universe = vanilla
    executable = /dataserver/users/formingworlds/postolec/miniconda3/bin/conda
    arguments = run --name proteus --no-capture-output proteus start --config /dataserver/users/formingworlds/postolec/PROTEUS/input/demos/escape.toml
    log = condor_outputs/log/logfile.$(PROCESS)
    output = condor_outputs/output/outfile.$(PROCESS)
    error = condor_outputs/output/errfile.$(PROCESS)
    notify_user = youremail@astro.rug.nl
    Requirements = (Cluster == "normas")
    queue 1
```

To exit nano, press `Ctrl+X`, then press `Enter` when prompted to save the file.

### Updating the Submit Script
Modify the following variables according to your job:

1. **`executable`**: Specify the absolute path to the Python environment (pyenv or  conda) you use to run PROTEUS. If you want to run another (python) script, you can modify the ```executable``` line with the absolute path to your script :
``` executable = /dataserver/users/formingworlds/lania/mscthesis/results/testscript.py```

2. **`arguments`**: Update the path to the config file for your PROTEUS simulation. If using `tools/grid_proteus.py`, modify the entire command accordingly. If you want to run another (python) script, you can modify the ```arguments``` line with the absolute path to your input and output directory : ``` arguments = -input [absolute path to input file] -outputdirectory [absolute path to output directory]```

3. **`notify_user`**: Enter your email address to receive job completion notifications.

- **`output`** : The outfile will contain the outputs/print statements of your job.
- **`error`** : The errfile file will contain the handled exceptions or runtime errors occuring while your job was running.

For further details, refer to the documentation on the Kapteyn intranet: [How to use Condor ?](https://www.astro.rug.nl/intranet/computing/index.php) (Go to Computing > Howto's > linux > How to use Condor ?)
This documentation is updated regularly, so be sure to check for the latest information. The HTCondor documentation can be found here [HT Condor manual](https://htcondor.readthedocs.io/en/latest/users-manual/submitting-a-job.html).

### Submitting and Monitoring Jobs
To submit your script, run: ```condor_submit name_of_your_script.submit```

To check the status of your job, use: ```condor_q``` or ```condor_q -better-analyze```.
The second command provides a more detailed job status analysis.

Another useful command is ```condor_status```. This displays the jobs currently running on Condormaster, including both your jobs and those of other users.

### Exiting Condormaster
To exit Condormaster and return to Norma1/Norma2, run: ```exit```

## Troubleshooting

### NetCDF Error

SOCRATES is using the NetCDF version installed by Python in your PROTEUS environment instead of the NetCDF version installed on the Kapteyn cluster system.

To resolve this issue:

1. Deactivate all conda environments.
2. Go to the PROTEUS folder : `cd PROTEUS/`
3. Delete the `socrates/` directory using `rm -r socrates/`
4. Run the `./tools/get_socrates.sh` command to download SOCRATES again, ensuring this is done OUTSIDE of any conda environment.
5. Execute the `cat socrates/set_rad_env` command to verify that SOCRATES is pointing to the correct NetCDF version (i.e. the NetCDF version installed on the Kapteyn cluster system).
6. Finally, run a PROTEUS simulation using the `default.toml` configuration file to confirm it is working correctly.

### Error reporting
- If you encounter an error that is not listed here, please create a new issue on the [PROTEUS GitHub webpage](https://github.com/FormingWorlds/PROTEUS/issues) (green button 'New issue' on the top right, choose 'Bug').
- Include details about what you were trying to do and how the error occurred. Providing a screenshot or copying/pasting the error message and log file can help others understand the issue better.
- Once the issue has been resolved, ensure that this troubleshooting section is updated to include the solution for future reference. You can check [here](./CONTRIBUTING.md) how to edit the documentation.
