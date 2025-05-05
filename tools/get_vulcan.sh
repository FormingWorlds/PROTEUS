#!/bin/bash
# Download and setup VULCAN

echo "Set up VULCAN..."

# Make room
workpath="VULCAN/"
rm -rf $workpath

use_ssh=false

# Download
echo "Cloning from GitHub"
if [ "$use_ssh" = true ]; then
    uri="git@github.com:FormingWorlds/VULCAN.git"
else
    uri="https://github.com/FormingWorlds/VULCAN.git"
fi
echo "    $uri -> $workpath"
git clone "$uri" "$workpath"

# Compile fastchem
# cd "$workpath/fastchem_vulcan/"
# make

# Done
echo "Done!"
