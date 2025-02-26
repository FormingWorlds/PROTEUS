#!/bin/bash
# Download and install PLATON

echo "Installing PLATON into Python environment..."

# Download dependencies
pip install astropy

# Make room
workpath="platon/"
rm -rf $workpath

# Download
git clone https://github.com/ideasrule/platon.git $workpath

# Change dir and install
olddir=$(pwd)
cd $workpath
pip install -e .
cd $olddir

# Import platon and run example - trigger data download
python "$workpath/examples/transit_depth_example.py"

echo "Done!"
