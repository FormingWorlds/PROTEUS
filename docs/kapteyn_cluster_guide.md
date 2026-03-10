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

    Finally, add the following entry in your `~/.ssh/config` file, making sure to add your username where appropriate.
    ```
    Host kapteyngateway
        HostName kapteyn.astro.rug.nl
        User YOUR_USERNAME_HERE
        IdentityFile ~/.ssh/id_rsa
        ForwardAgent yes

    Host norma2
        HostName norma2
        User YOUR_USERNAME_HERE
        IdentityFile ~/.ssh/id_rsa
        ProxyJump kapteyngateway
        ServerAliveInterval 120
        ServerAliveCountMax 60
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

4. To avoid the cluster terminating PROTEUS jobs, increase the temporary file limit for your user by adding to your shell rc file (e.g., '~/.bashrc'):
    ```console
    echo "ulimit -Sn 4000000" >> "$HOME/.bashrc"
    echo "ulimit -Hn 5000000" >> "$HOME/.bashrc"
    ```
    Then, reload your shell rc file to make the changes effective:
    ```console
    source "$HOME/.bashrc"
    ```

5. You can now follow the usual installation steps [here](installation.html), but, since your home folder is capped
   at 9GB, you need to install Julia and miniconda or conda-forge in "/dataserver/users/formingworlds/<username>".
    ### Julia considerations
    If you have already installed Julia in your home folder, you could remove that through `rm -rf ~/.julia`.
 
    If you install Julia through Juliaup this involves:
    ```console
    export JULIAUP_HOME=/dataserver/users/formingworlds/<username>/.juliaup
    curl -fsSL https://install.julialang.org | sh
    ```

    To also make sure that the Julia ecosystem, such as Julia packages, are also not installed in `$HOME`, add `JULIA_DEPOT_PATH` to your `~/.shellrc`, e.g. `~/.bashrc`:
    ```console
    export JULIA_DEPOT_PATH=/dataserver/users/formingworlds/<username>/.julia
    ```
    Setting only this variable will be sufficient if you have not installed Julia through Juliaup.
    In any case, it is best to have both of these Julia environment variables exported when you log in,
    so please add this to your `~/.shellrc`, e.g. `~/.bashrc`:
    ```console
    export JULIAUP_HOME=/dataserver/users/formingworlds/<username>/.juliaup
    export JULIA_DEPOT_PATH="/dataserver/users/formingworlds/<username>/.julia"
    ```
    If you install Julia using `tar`, use the following steps:

   ```
    export JULIA_DIR=/dataserver/users/formingworlds/<username>/julia-1.11.6
    
    mkdir -p $JULIA_DIR
    
    cd /dataserver/users/formingworlds/<username>
    
    wget https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.6-linux-x86_64.tar.gz
    
    tar -xvzf julia-1.11.6-linux-x86_64.tar.gz
    
    echo 'export PATH=/dataserver/users/formingworlds/<username>/julia-1.11.6/bin:$PATH' >> ~/.bashrc
    
    echo 'export JULIA_DEPOT_PATH=/dataserver/users/formingworlds/<username>/.julia' >> ~/.bashrc
    
    source ~/.bashrc
   ```
  
    ### Miniconda and conda-forge considerations
    When installing miniconda or conda-forge, make sure you do not choose the default path, which is always your home folder. Adjust it to `/dataserver/users/formingworlds/<username>`.
    Alternatively, you can set default paths upfront for miniconda:
    ```console
    mkdir -p /dataserver/users/formingworlds/<username>/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O 
        /dataserver/users/formingworlds/<username>/miniconda3/miniconda.sh
    bash /dataserver/users/formingworlds/<username>/miniconda3/miniconda.sh -b -u -p 
        /dataserver/users/formingworlds/<username>/miniconda3
    rm /dataserver/users/formingworlds/<username>/miniconda3/miniconda.sh
    ```
    and similarly for conda-forge:
    ```console
    mkdir -p /dataserver/users/formingworlds/${USER}/miniforge3
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -O 
        /dataserver/users/formingworlds/<username>/miniforge3/miniforge.sh
    bash /dataserver/users/formingworlds/<username>/miniforge3/miniforge.sh -b -p 
        /dataserver/users/formingworlds/<username>/miniforge3
    rm /dataserver/users/formingworlds/<username>/miniforge3/miniforge.sh
    ``` 
    For both Miniconda and conda-forge follow the instructions wrt updating your `~/.shellrc` file.

    ### Pip cache consideration
    The pip cache can easily take more than 3 GB when installing PROTEUS and this may exceed your 
    disk quota on your home directory. Therefore, you need to setup your pip cache folder in a different
    place:
    ```console
    mkdir /dataserver/users/formingworlds/${USER}/.pip-cache
    export PIP_CACHE_DIR=/dataserver/users/formingworlds/${USER}/.pip-cache 
    ```
    
## Queuing Manager: Condormaster

- To use the queuing manager on the Kapteyn cluster, you first need to SSH into Norma1 or Norma2.
    ```console
    ssh norma1
    ```

- To access Condormaster, run the following command :
    ```console
    ssh condormaster
    ```

### Submitting a Job on Condormaster
- To run a job using Condormaster, you first need to write a submit script. Begin by navigating to your home directory and creating a new submit script using :
    ```console
    nano name_of_your_script.submit
    ```

- You can copy and paste the example submit script below (to start a single PROTEUS simulation) and modify it according to your needs.

```console
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
Modify the following variables according to your needs :

- **`executable`**: Specify the absolute path to the Python environment (pyenv or  conda) you use to run PROTEUS. If you want to run another (python) script, you can modify the ```executable``` line with the absolute path to your script :

``` executable = /dataserver/users/formingworlds/lania/mscthesis/results/testscript.py```

- **`arguments`**: Update the path to the config file for your PROTEUS simulation. If using `tools/grid_proteus.py`, modify the entire command accordingly. If you want to run another (python) script, you can modify the ```arguments``` line with the absolute path to your input and output directory :

``` arguments = -input [absolute path to input file] -outputdirectory [absolute path to output directory]```

- **`notify_user`**: Enter your email address to receive job completion notifications.

- **`output`** : The outfile will contain the outputs/print statements of your job.

- **`error`** : The errfile file will contain the handled exceptions or runtime errors occuring while your job was running.

For further details, refer to the documentation on the Kapteyn intranet: [How to use Condor?](https://www.astro.rug.nl/intranet/computing/index.php) (Go to Computing > Howto's > linux > How to use Condor?)
This documentation is updated regularly, so be sure to check for the latest information. Also for more details about condor, the HTCondor documentation can be found here [HT Condor manual](https://htcondor.readthedocs.io/en/latest/users-manual/submitting-a-job.html).

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
- Once the issue has been resolved, ensure that this troubleshooting section is updated to include the solution for future reference. You can check [here](CONTRIBUTING.html) how to edit the documentation.
