# Tips & Tricks to Run PROTEUS on the Kapteyn Cluster

## Installation

1. Connect via SSH. Manage your authentication keys to avoid entering your password every time you connect (optional). You can find the instructions on the Kapteyn intranet: [How to generate authentication keys for SSH, SFTP, and SCP](https://www.astro.rug.nl/intranet/computing/index.php) (Go to Computing > Howto's > How to generate authentication keys for ssh, sftp and scp).

2. Follow the installation instructions [here](./installation.md).

3. Increase the file limit for your user by adding to your shell rc file: `ulimit -Sn 4000000` and `ulimit -Hn 5000000`.

## Tricks

- To launch a simulation on the Kapteyn cluster, avoid using `screen` sessions (it behaves inconsistently on the cluster). Instead, use the similar tool `tmux`. You can find detailed documentation [here](https://tmuxcheatsheet.com/).

## Queuing Manager : Condormaster

To use the queuing manager on the Kapteyn cluster, you first need to SSH into Norma1 or Norma2.

To access Condormaster, run the following command:

```ssh condormaster```


### Submitting a Job on Condormaster
To run a job using Condormaster, you first need to write a submit script. Begin by navigating to your home directory and creating a new submit script using:

```nano name_of_your_script.submit```

You can copy and paste the example submit script below and modify it according to your needs.

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
1. **`executable`**: Specify the path to the Python environment you use to run PROTEUS.
2. **`arguments`**: Update the path to the config file for your PROTEUS simulation. If using `tools/grid_proteus.py`, modify the entire command accordingly.
3. **`notify_user`**: Enter your email address to receive job completion notifications.

For further details, refer to the documentation on the Kapteyn intranet: [How to use Condor?] (https://www.astro.rug.nl/intranet/computing/index.php)
This documentation is updated regularly, so be sure to check for the latest information.

### Submitting and Monitoring Jobs
To submit your script, run:

```condor submit name_of_your_script.submit```

To check the status of your job, use:

```condor_q```
or
```condor_q -better-analyze```

The second command provides a more detailed job status analysis.

Another useful command is:

```condor_status```

This displays the jobs currently running on Condormaster, including both your jobs and those of other users.

### Exiting Condormaster
To exit Condormaster and return to Norma1/Norma2, run:

```exit```

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
- If you encounter an error that is not listed here, please create a new issue on the [PROTEUS GitHub webpage](https://github.com/FormingWorlds/PROTEUS/issues) (green button 'New issue' on the top right). Include details about what you were trying to do and how the error occurred. Providing a screenshot or copying/pasting the error message can help others understand the issue better. Once the issue has been resolved, ensure that this troubleshooting section is updated to include the solution for future reference.
