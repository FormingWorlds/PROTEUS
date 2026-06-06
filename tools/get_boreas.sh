#!/bin/bash
# Download and setup BOREAS (optional hydrodynamic escape module)

echo "Set up BOREAS..."

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
workpath=$root/BOREAS/
rm -rf $workpath

# Check SSH access to GitHub (exit code 1 = authenticated). The command
# sits inside the if so a future `set -e` cannot abort the script here.
if ssh -T git@github.com; then
    use_ssh=false
else
    if [ $? -eq 1 ]; then
        use_ssh=true
    else
        use_ssh=false
    fi
fi

# Resolve the pinned URL + ref from pyproject.toml.
b_url=$(python "$root/tools/_module_pins.py" boreas url)
b_ref=$(python "$root/tools/_module_pins.py" boreas ref)

echo "Cloning from GitHub"
if [ "$use_ssh" = true ]; then
    # Rewrite https://github.com/ -> git@github.com: for SSH transport.
    uri=${b_url/https:\/\/github.com\//git@github.com:}
else
    uri="$b_url"
fi
echo "    $uri @ $b_ref -> $workpath"
git clone "$uri" "$workpath"
git -C "$workpath" checkout --quiet "$b_ref"

# Install boreas package
pip install -U -e "$workpath"

# Done
echo "Done!"
