#!/bin/bash
# Download and compile spider
# Pass folder as argument to use that as the download path

set -e

# Check environment
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PETSC_ARCH=arch-linux-c-opt
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PETSC_ARCH=arch-darwin-c-opt
else
    echo "ERROR: Unknown OS type '$OSTYPE' "
fi

# PETSc install folder
PETSC_DIR=$(realpath petsc)

# Export if needed
export PETSC_ARCH
export PETSC_DIR

echo "Using PETSC_DIR=$PETSC_DIR"
echo "Using PETSC_ARCH=$PETSC_ARCH"

# Output path
workpath="SPIDER"
if [ -n "$1" ]; then
    workpath=$1
fi
rm -rf $workpath
mkdir $workpath

# Download zip file
zip="$workpath/spider.zip"
url="https://osf.io/download/s8gb9/"
echo "Downloading archive file from OSF"
echo "    $url -> $zip"
sleep 1
curl -LsS $url > $zip

# Decompress zip file
echo "Decompressing"
unzip -qq $zip -d $workpath
rm $zip

# Change dir
olddir=$(pwd)
cd $workpath

# Make
echo "Make"
make clean
make -j

# Test
# ./spider -options_file tests/opts/blackbody50.opts

# Change dir
cd $olddir
echo "Done"
