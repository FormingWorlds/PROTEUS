#!/bin/bash
# Download and compile spider
# Pass folder as argument to use that as the download path

set -e

# Check environment
if [ -z $PETSC_ARCH ]; then
    echo "ERROR: You need to set PETSC_ARCH before compiling SPIDER"
    exit 1
fi
if [ -z $PETSC_DIR ]; then
    echo "ERROR: You need to set PETSC_DIR before compiling SPIDER"
    exit 1
fi

# Output path
workpath="spider"
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
