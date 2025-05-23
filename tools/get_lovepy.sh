#!/bin/bash
# Download and install LovePy

# Julia installed?
if ! [ -x "$(command -v julia)" ]; then
  echo 'ERROR: Julia is not installed.' >&2
  exit 1
fi

# Install LovePy package
echo "Installing LovePy into Julia environment..."
LD_LIBRARY_PATH="" julia -e 'using Pkg; Pkg.add(url="https://github.com/nichollsh/LovePy")'
LD_LIBRARY_PATH="" julia -e 'using LovePy; println("Installed to: "*pathof(LovePy))'

echo "Done!"
