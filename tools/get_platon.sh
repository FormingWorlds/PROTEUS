#!/bin/bash
# Download and install PLATON

echo "Installing PLATON into Python environment..."

# Make room
workpath="platon/"
rm -rf $workpath

# Download
echo "Cloning from GitHub"
if [ "$use_ssh" = true ]; then
    uri="git@github.com:nichollsh/platon.git"
else
    uri="https://github.com/nichollsh/platon.git"
fi
echo "    $uri -> $workpath"
git clone "$uri" "$workpath"

# Change dir and install
olddir=$(pwd)
cd $workpath
python -m pip install -U -e .
cd $olddir

# Import platon and run example - trigger data download
python "$workpath/examples/transit_depth_example.py"

echo "Done!!"
