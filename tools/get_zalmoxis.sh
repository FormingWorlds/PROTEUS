#!/bin/bash
# Download and setup Zalmoxis as an editable sibling checkout.
#
# Clones FormingWorlds/Zalmoxis into ./Zalmoxis/ inside the PROTEUS
# root and installs it editable into the active Python environment.
# The editable install takes precedence over the PyPI fwl-zalmoxis
# pin in pyproject.toml on sys.path, so any local edits to
# Zalmoxis/src/ are picked up by `import zalmoxis` without
# reinstalling.

echo "Set up Zalmoxis..."

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
workpath=$root/Zalmoxis/
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
    uri="git@github.com:FormingWorlds/Zalmoxis.git"
else
    uri="https://github.com/FormingWorlds/Zalmoxis.git"
fi
echo "    $uri -> $workpath"
git clone "$uri" "$workpath" || { echo "ERROR: git clone failed" >&2; exit 1; }

# Install zalmoxis package as editable
cd "$workpath" || { echo "ERROR: cannot enter $workpath" >&2; exit 1; }
pip install -U -e . || { echo "ERROR: editable install failed" >&2; exit 1; }

# Back to old folder
cd $root

# Done
echo "Done!"
