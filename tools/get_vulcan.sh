#!/bin/bash
# Download and setup VULCAN

echo "Set up VULCAN..."

# Make room
workpath="VULCAN/"
rm -rf $workpath

# Download
echo "Cloning from GitHub"
if [ "$use_ssh" = true ]; then
    uri="git@github.com:nichollsh/VULCAN.git"
else
    uri="https://github.com/nichollsh/VULCAN.git"
fi
echo "    $uri -> $workpath"
git clone "$uri" "$workpath"

# Compile fastchem
# cd "$workpath/fastchem_vulcan/"
# make

# Done
echo "Done!"
