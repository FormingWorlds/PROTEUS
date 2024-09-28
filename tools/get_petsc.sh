#!/bin/bash
# Download and compile petsc
# Pass folder as argument to use that as the download path

set -e

# Output path
workpath="petsc"
if [ -n "$1" ]; then
    workpath=$1
fi
workpath=$(realpath $workpath) # petsc doesn't like relative paths
rm -rf $workpath
mkdir $workpath

# Download zip file
zip="$workpath/petsc.zip"
url="https://osf.io/download/p5vxq/"
echo "Downloading archive file from OSF"
echo "    $url -> $zip"
sleep 1
curl -LsS $url > $zip

# Decompress zip file
echo "Decompressing"
unzip -qq $zip -d $workpath
rm $zip

# Set environment
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export PETSC_ARCH=arch-linux-c-opt
elif [[ "$OSTYPE" == "darwin"* ]]; then
    export PETSC_ARCH=arch-darwin-c-opt
else
    echo "ERROR: Unknown OS type '$OSTYPE' "
    exit 1
fi
export PETSC_DIR=$workpath

# Change dir
olddir=$(pwd)
cd $workpath

# Configure
echo "Configure"
./configure --with-debugging=0 --with-fc=0 --with-cxx=0 --download-sundials2 --download-mpich --download-f2cblaslapack --COPTFLAGS="-g -O3" --CXXOPTFLAGS="-g -O3"

# Build
echo "Build"
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH all

# Test
echo "Test"
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH check

# Change dir
cd $olddir
echo "Done"
