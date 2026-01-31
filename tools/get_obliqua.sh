#!/bin/bash
# Download and install Obliqua

# Julia installed?
if ! [ -x "$(command -v julia)" ]; then
  echo 'ERROR: Julia is not installed.' >&2
  exit 1
fi

# Install Obliqua (Obliqua.jl) into Julia environment
echo "Installing Obliqua into Julia environment..."

LD_LIBRARY_PATH="" julia -e '
using Pkg
Pkg.add(url="https://github.com/FormingWorlds/Obliqua", rev="md/...")
'

LD_LIBRARY_PATH="" julia -e '
using Obliqua
println("Installed to: " * pathof(Obliqua))
'

echo "Done!"
