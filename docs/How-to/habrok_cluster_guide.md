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

3. You can now follow the usual installation steps [here](installation.md).

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

See the section on running grids in the PROTEUS [usage guide](usage.md#running-grids-of-simulations-ensembles). These instructions will detail how to submit grids to the nodes via SLURM.

You can also submit single PROTEUS runs to the nodes. For example:
```console
sbatch --mem-per-cpu=3G --time=1440 --wrap "proteus start -oc input/all_options.toml"
```

## Transferring data from Habrok to Kapteyn

Habrok and Kapteyn are on different networks. Habrok cannot reach Kapteyn (the firewall blocks outgoing SSH), and although Kapteyn can reach Habrok, Habrok requires two-factor authentication (2FA) for every connection, which makes automated transfers from Kapteyn difficult.

So you cannot simply run `rsync` or `scp` in either direction between the two clusters. The workaround is to relay data through a machine that can reach both, like your laptop:

```
Habrok  -->  your laptop  -->  Kapteyn (norma2)
         pull                push
```

### Prerequisites

You need SSH access to both clusters configured on your laptop. See the [Habrok SSH setup](#access-the-habrok-cluster) above and the [Kapteyn cluster guide](kapteyn_cluster_guide.md) for SSH config instructions, including the ProxyJump setup needed to reach `norma2`.

Test that both connections work before proceeding:

```console
ssh habrok    # will ask for your TOTP code
ssh norma2    # key-based, no 2FA
```

### Step 1: Pull data from Habrok to your laptop

On Habrok, PROTEUS output typically lives in `/scratch/<habrok_user>/proteus_output/`. Check what is there:

```console
ssh habrok 'ls -lh /scratch/<habrok_user>/proteus_output/'
```

Pull it to a temporary folder on your laptop:

```console
mkdir -p /tmp/habrok_transfer
rsync -avz habrok:/scratch/<habrok_user>/proteus_output/my_run/ /tmp/habrok_transfer/my_run/
```

Replace `<habrok_user>` with your Habrok username (e.g., `p000000`) and `my_run` with your simulation directory name.

If you only need the CSV and plots (not the raw per-timestep data), add `--exclude=data/` to save time and disk space:

```console
rsync -avz --exclude=data/ habrok:/scratch/<habrok_user>/proteus_output/my_run/ /tmp/habrok_transfer/my_run/
```

### Step 2: Push data from your laptop to Kapteyn

Push the staged data to the Kapteyn dataserver:

```console
ssh norma2 'mkdir -p /dataserver/users/formingworlds/<kapteyn_user>/proteus_output/my_run'
rsync -avz /tmp/habrok_transfer/my_run/ norma2:/dataserver/users/formingworlds/<kapteyn_user>/proteus_output/my_run/
```

Replace `<kapteyn_user>` with your Kapteyn username.

### Step 3: Clean up

Remove the temporary staging data from your laptop:

```console
rm -rf /tmp/habrok_transfer/my_run
```

### Alternative: direct pipe (no staging on your laptop)

Instead of storing data on your laptop in between, you can pipe the data straight through in a single command using SSH and `tar`:

```console
ssh habrok 'tar -cf - -C /scratch/<habrok_user>/proteus_output my_run' \
  | ssh norma2 'tar -xf - -C /dataserver/users/formingworlds/<kapteyn_user>/proteus_output'
```

This streams data from Habrok through your laptop to Kapteyn without writing anything to disk locally. The downside is that if the connection drops, you have to start over from scratch (unlike `rsync`, which can resume). This approach is best for smaller transfers.

To exclude the `data/` directory (slim transfer):

```console
ssh habrok 'tar -cf - --exclude=data -C /scratch/<habrok_user>/proteus_output my_run' \
  | ssh norma2 'tar -xf - -C /dataserver/users/formingworlds/<kapteyn_user>/proteus_output'
```

### Tips

- **rsync is incremental.** If the transfer gets interrupted (laptop goes to sleep, WiFi drops), re-run the same `rsync` command. It picks up where it left off and only transfers new or changed files.
- **Check sizes first.** Before pulling, check how large the data is: `ssh habrok 'du -sh /scratch/<habrok_user>/proteus_output/my_run/'`. Large runs can be tens of GB.
- **The `data/` directory is often not needed.** It contains raw NetCDF/JSON output at every timestep. The `runtime_helpfile.csv` and `plots/` directory are usually sufficient for analysis.
- **Kapteyn storage quotas.** The formingworlds dataserver has also limited space. Check your usage with `ssh norma2 'du -sh /dataserver/users/formingworlds/<kapteyn_user>/'` before transferring large datasets.
