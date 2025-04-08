#!/bin/bash
# Download and setup VULCAN

echo "Set up VULCAN..."

# Make room
workpath="VULCAN/"
rm -rf $workpath

# Download
git clone git@github.com:nichollsh/VULCAN.git $workpath

# Compile fastchem
# cd "$workpath/fastchem_vulcan/"
# make

# Done
echo "Done!"
