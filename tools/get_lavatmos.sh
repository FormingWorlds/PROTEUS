#!/bin/bash
# Download lavatmos

set -e

root=$(dirname $(realpath $0))
root=$(realpath "$root/..")

# Download via HTTPS only
lavapath="$root/Thermoengine/LavAtmos"
rm -rf "$lavapath"

git clone https://github.com/leojola/LavAtmos.git "$lavapath"


cd "$lavapath"
cd $root
export LAVATMOS_DIR=$lavapath

echo "LavAtmos has been installed"
echo "It is recommended that you add the following line to your shell rc file"
echo "export LAVATMOS_DIR='$lavapath'"
exit 0
