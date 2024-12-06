# Tips & Tricks to Run PROTEUS on the Kapteyn Cluster

## Installation

1. Connect via SSH. Manage your authentication keys to avoid entering your password every time you connect (optional). You can find the instructions on the Kapteyn intranet: [How to generate authentication keys for SSH, SFTP, and SCP](https://www.astro.rug.nl/intranet/computing/index.php) (Go to Computing > Howto's > How to generate authentication keys for ssh, sftp and scp).

2. Follow the installation instructions [here](./installation.md).

## Tricks

- To launch a simulation on the Kapteyn cluster, avoid using `screen` sessions (it behaves inconsistently on the cluster). Instead, use the similar tool `tmux`. You can find detailed documentation [here](https://tmuxcheatsheet.com/).

## Troubleshooting

### NetCDF Error:

    SOCRATES is using the NetCDF version installed by Python in your PROTEUS environment instead of the NetCDF version installed on the Kapteyn cluster system.

    To resolve this issue:

    1. Deactivate all conda environments.
    2. Go to the PROTEUS folder : `cd PROTEUS/`
    3. Delete the `socrates/` directory using `rm -r socrates/`
    4. Run the `./tools/get_socrates.sh` command to download SOCRATES again, ensuring this is done OUTSIDE of any conda environment.
    5. Execute the `cat socrates/set_rad_env` command to verify that SOCRATES is pointing to the correct NetCDF version (i.e. the NetCDF version installed on the Kapteyn cluster system).
    6. Finally, run a PROTEUS simulation using the `default.toml` configuration file to confirm it is working correctly.

### Error reporting
- If you encounter a new error not listed here, please document it in this shared [GoogleDoc] (https://docs.google.com/document/d/13YpMmWX8ru0PKb4UuzByX86s1PYG3OPIhABcT0kv4zY/edit?usp=sharing) so that others can view and collaborate on resolving it. Once the issue is resolved, make sure to update the troubleshooting section of this documentation to reflect the solution.