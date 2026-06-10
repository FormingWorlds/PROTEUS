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

portable_realpath() {
    if command -v realpath >/dev/null 2>&1; then
        realpath "$1"
    else
        python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "$1"
    fi
}


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
root=$(dirname $(portable_realpath $0))
root=$(portable_realpath "$root/..")

# Separate the --force flag from the optional install-path argument.
force=false
install_path=""
for arg in "$@"; do
    if [ "$arg" = "--force" ]; then
        force=true
    elif [ -z "$install_path" ]; then
        install_path="$arg"
    fi
done

if [ -n "$install_path" ]; then
    socpath="$(portable_realpath "$install_path")"
else
    socpath="$root/socrates"
fi

# Refuse to delete a checkout holding local work unless --force is given.
# Keep this guard in sync across the get_* scripts that refresh checkouts.
# Guarded states: modified tracked files, and commits not on any remote.
# Untracked files (the compiled build tree) do not block the refresh.
# make/Mk_cmd is excluded: configure regenerates it on every build (compiler
# detection, host paths, and optimisation flags), so it is regenerable build
# config rather than user work and would otherwise block every refresh.
if [ -d "$socpath/.git" ] && [ "$force" != true ]; then
    dirty=$(git -C "$socpath" status --porcelain --untracked-files=no \
        -- ':(exclude)make/Mk_cmd' 2>/dev/null | head -1)
    unpushed=$(git -C "$socpath" log HEAD --not --remotes --oneline 2>/dev/null | head -1)
    if [ -n "$dirty" ] || [ -n "$unpushed" ]; then
        echo "ERROR: $socpath has uncommitted changes or commits not on a remote." >&2
        echo "       Refusing to delete it. Commit and push your work, or run" >&2
        echo "       bash tools/get_socrates.sh --force  to discard the checkout." >&2
        exit 1
    fi
fi
rm -rf "$socpath"

set -euo pipefail


# Resolve the pinned URL + ref from pyproject.toml. The HTTPS URL is the
# default; SSH is used only when ssh -T against github succeeded above.
soc_url=$(python "$root/tools/_module_pins.py" socrates url)
soc_ref=$(python "$root/tools/_module_pins.py" socrates ref)

if [ "$use_ssh" = true ]; then
    # Rewrite https://github.com/ -> git@github.com: for SSH transport.
    soc_ssh_url=${soc_url/https:\/\/github.com\//git@github.com:}
    git clone "$soc_ssh_url" "$socpath"
else
    git clone "$soc_url" "$socpath"
fi

# Pin to the configured SHA / tag / branch.
git -C "$socpath" checkout --quiet "$soc_ref"

# Compile SOCRATES
cd "$socpath"
./configure

# Compile against a portable instruction set. The SOCRATES configure
# script sets the optimisation flags to "-Ofast -march=native", which
# bakes the build host's specific CPU extensions into the binaries. Such
# a binary aborts with an illegal-instruction fault if the compiled tree
# is reused on a host whose CPU lacks those extensions, which happens
# whenever the build is cached and restored across machines with
# different processors. Rewrite the generated flags to "-O2
# -fno-fast-math", which runs on any CPU and avoids fast-math value
# reordering.
if grep -q -- '-Ofast -march=native' make/Mk_cmd; then
    sed 's/-Ofast -march=native/-O2 -fno-fast-math/g' make/Mk_cmd > make/Mk_cmd.portable
    mv make/Mk_cmd.portable make/Mk_cmd
fi

# Stop the build if any non-portable optimisation flag survived the rewrite.
# The substitution above keys on the exact flag string configure writes
# today; a future SOCRATES release that changes that string would otherwise
# skip the rewrite and silently compile a host-specific binary that aborts
# when the cached build is reused on a different CPU. Fail here instead.
if grep -qE -- '-march=native|-Ofast' make/Mk_cmd; then
    echo "ERROR: non-portable optimisation flags remain in make/Mk_cmd." >&2
    echo "       The SOCRATES configure flag string has changed; update the" >&2
    echo "       rewrite in tools/get_socrates.sh to match." >&2
    grep -nE -- '-march=native|-Ofast' make/Mk_cmd >&2
    exit 1
fi

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
