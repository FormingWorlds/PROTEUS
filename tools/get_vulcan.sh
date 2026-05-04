#!/bin/bash
# Download and setup VULCAN

echo "Set up VULCAN..."

portable_realpath() {
    if command -v realpath >/dev/null 2>&1; then
        realpath "$1"
    else
        python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "$1"
    fi
}

# Path to PROTEUS folder
root=$(dirname $(portable_realpath $0))
root=$(portable_realpath "$root/..")

# Make room
workpath=$root/VULCAN/
rm -rf $workpath

# Check SSH access to GitHub
ssh -T git@github.com
if [ $? -eq 1 ]; then
    use_ssh=true
else
    use_ssh=false
fi

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
cd "$workpath/fastchem_vulcan/"
make
cd $workpath

# Install vulcan package
pip install -U -e .

# Back to old folder
cd $root

# Done
echo "Done!"
