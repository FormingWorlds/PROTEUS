#!/bin/bash
# Download and compile socrates
# Pass folder as argument to use that as the download path

# Check SSH access to GitHub
ssh -T git@github.com
if [ $? -eq 1 ]; then
    use_ssh=true
else
    use_ssh=false
fi
set -e

# Output path
socpath="socrates"
if [ -n "$1" ]; then
    socpath=$1
fi
rm -rf "$socpath"

# Download (using SSH if possible)
echo "Cloning from GitHub"
if [ "$use_ssh" = true ]; then
    uri="git@github.com:nichollsh/SOCRATES.git"
else
    uri="https://github.com/nichollsh/SOCRATES.git"
fi
echo "    $uri -> $socpath"
git clone "$uri" "$socpath"

# Configure and build SOCRATES
olddir=$(pwd)
cd "$socpath"
./configure
./build_code

# Inform user
echo "SOCRATES has been downloaded and built in $socpath"
echo "Now add the following to your `~/.bashrc` file:"
echo "    export RAD_DIR=$socpath"

# Environment
# source ./set_rad_env
# export LD_LIBRARY_PATH=""
cd $olddir
