#!/bin/bash
# Download and compile ThermoEngineLite

# Do we have clang?
if ! [ -x "$(command -v clang)" ]; then
  echo 'ERROR: clang is not installed.' >&2
  exit 1
fi

# Do we have pip?
if ! [ -x "$(command -v pip)" ]; then
  echo 'ERROR: pip is not installed.' >&2
  exit 1
fi

# Shared helpers: see tools/_get_common.sh.
_get_common="$(dirname "${BASH_SOURCE[0]}")/_get_common.sh"
if [ ! -f "$_get_common" ]; then
    echo "ERROR: $_get_common is missing; use a complete PROTEUS checkout." >&2
    exit 1
fi
source "$_get_common"

# Path to PROTEUS folder
root="$proteus_root"

get_parse_args "$@"
workpath="$root/ThermoEngineLite/"

guard_dirty_checkout "$workpath" get_thermoenginelite.sh

# Make room
rm -rf "$workpath"

# Detect SSH access to GitHub.
use_ssh=$(github_use_ssh)

# Resolve the pinned URL + ref from pyproject.toml.
resolve_module_pin thermoenginelite
l_url="$module_url"
l_ref="$module_ref"

echo "Cloning from GitHub"
if [ "$use_ssh" = true ]; then
    uri=$(github_ssh_url "$l_url")
else
    uri="$l_url"
fi
echo "    $uri @ $l_ref -> $workpath"
git clone "$uri" "$workpath" || { echo "ERROR: git clone failed" >&2; exit 1; }
git -C "$workpath" checkout --quiet "$l_ref" \
    || { echo "ERROR: cannot checkout $l_ref" >&2; exit 1; }

# Compile ThermoEngine
echo "Compiling ThermoEngineLite..."
echo "    This will take ~20 minutes to complete"
cd "$workpath"
make devinstall


# Check that the library was and installed into python environment
if ! python -c "import thermoengine" >/dev/null 2>&1; then
    echo "ERROR: ThermoEngineLite failed to install into your Python environment." >&2
    echo "       Check the output above for errors." >&2
    exit 1
fi
