#!/bin/bash
# Download and compile socrates

# Do we have NetCDF?
if ! [ -x "$(command -v nc-config)" ]; then
  echo 'ERROR: NetCDF is not installed.' >&2
  exit 1
fi
if ! [ -x "$(command -v nf-config)" ]; then
  echo 'ERROR: NetCDF-Fortran library is not installed.' >&2
  exit 1
fi

# Do we have gfortran?
if ! [ -x "$(command -v gfortran)" ]; then
  echo 'ERROR: gfortran compiler is not installed.' >&2
  exit 1
fi

# Already setup?
if [ -n "$RAD_DIR" ]; then
    echo "WARNING: You already have SOCRATES installed"
    echo "         RAD_DIR=$RAD_DIR"
    echo "Reinstalling SOCRATES..."
    echo ""
    sleep 5
fi

# Check SSH access to GitHub
ssh -T git@github.com
if [ $? -eq 1 ]; then
    use_ssh=true
else
    use_ssh=false
fi

# Disable SSH (uncomment to allow SSH clone of SOCRATES)
# use_ssh=false

# Download
root=$(dirname $(realpath $0))
root=$(realpath "$root/..")
socpath="$root/socrates"
rm -rf "$socpath"
if [ "$use_ssh" = true ]; then
    git clone git@github.com:FormingWorlds/SOCRATES.git "$socpath"
else
    git clone https://github.com/FormingWorlds/SOCRATES.git "$socpath"
fi

# Compile SOCRATES
cd "$socpath"
./configure
./build_code

# Environment
export RAD_DIR=$socpath
cd $root

# Check radlib exists
radlib="$socpath/bin/radlib.a"
if [ -f "$radlib" ]; then
    echo "SOCRATES has been installed"
    echo ""
else
    echo "Could not find compiled SOCRATES binaries - failed to compile"
    exit 1
fi


# Inform user
echo "You must now run the following command:"
echo "    export RAD_DIR='$socpath'"
echo " "
echo "You should also add this command to your shell rc file (e.g. ~/.bashrc)"
exit 0
