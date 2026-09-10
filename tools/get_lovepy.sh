#!/bin/bash
# Download and install LovePy

# Julia installed?
if ! [ -x "$(command -v julia)" ]; then
  echo 'ERROR: Julia is not installed.' >&2
  exit 1
fi

# Shared helpers: see tools/_get_common.sh.
_get_common="$(dirname "${BASH_SOURCE[0]}")/_get_common.sh"
if [ ! -f "$_get_common" ]; then
    echo "ERROR: $_get_common is missing; use a complete PROTEUS checkout." >&2
    exit 1
fi
source "$_get_common"

# Resolve the pinned LovePy URL + ref from pyproject.toml. Julia's Pkg.add
# accepts a `rev=` kwarg for a specific commit / tag / branch.
resolve_module_pin lovepy
lp_url="$module_url"
lp_ref="$module_ref"

echo "Installing LovePy into Julia environment ($lp_url @ $lp_ref)..."
LD_LIBRARY_PATH="" julia -e "using Pkg; Pkg.add(url=\"$lp_url\", rev=\"$lp_ref\")"
LD_LIBRARY_PATH="" julia -e 'using LovePy; println("Installed to: "*pathof(LovePy))'

echo "Done!"
