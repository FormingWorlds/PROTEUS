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

# Output path
socpath="socrates"
if [ -n "$1" ]; then
    socpath=$1
fi

# Download (using SSH if possible)
rm -rf "$socpath"
if [ "$use_ssh" = true ]; then
    git clone git@github.com:nichollsh/SOCRATES.git "$socpath"
else
    git clone https://github.com/nichollsh/SOCRATES.git "$socpath"
fi

# Compile SOCRATES
cd "$socpath"
./configure
./build_code

# Environment
source ./set_rad_env
export LD_LIBRARY_PATH=""
cd ..
