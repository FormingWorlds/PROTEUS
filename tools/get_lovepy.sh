#!/bin/bash
# Download and install LovePy

# Julia installed?
if ! [ -x "$(command -v julia)" ]; then
  echo 'ERROR: Julia is not installed.' >&2
  exit 1
fi

# Resolve the pinned LovePy URL + ref from pyproject.toml. Julia's
# Pkg.add accepts a `rev=` kwarg for a specific commit / tag / branch;
# `main` is the default if no ref is configured.
script_root="$(cd "$(dirname "$0")/.." && pwd)"
lp_url=$(python "$script_root/tools/_module_pins.py" lovepy url)
lp_ref=$(python "$script_root/tools/_module_pins.py" lovepy ref)

echo "Installing LovePy into Julia environment ($lp_url @ $lp_ref)..."
LD_LIBRARY_PATH="" julia -e "using Pkg; Pkg.add(url=\"$lp_url\", rev=\"$lp_ref\")"
LD_LIBRARY_PATH="" julia -e 'using LovePy; println("Installed to: "*pathof(LovePy))'

echo "Done!"
