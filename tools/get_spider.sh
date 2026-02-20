#!/usr/bin/env bash
# =============================================================================
# get_spider.sh — Download and compile SPIDER for PROTEUS
# =============================================================================
#
# Clones the SPIDER interior evolution model from GitHub and compiles it
# against a local PETSc installation. PETSc must be installed first
# (via get_petsc.sh).
#
# SPIDER is a pure C code that uses PETSc for numerics and sundials2 for
# ODE integration. The Makefile includes PETSc's build rules, so PETSC_DIR
# and PETSC_ARCH must be set correctly.
#
# Supported platforms:
#   - macOS 10.15 (Catalina) and later, Intel and Apple Silicon
#   - Linux (Ubuntu, Debian, Fedora/RHEL, HPC clusters)
#
# Prerequisites:
#   - PETSc built via ./tools/get_petsc.sh (must complete first)
#   - C compiler accessible via MPI wrapper (mpicc)
#   - git (for cloning the repository)
#
# Usage:
#   ./tools/get_spider.sh              # clone into ./SPIDER/
#   ./tools/get_spider.sh /path/to/dir # clone into specified directory
#
# The script is also called programmatically by proteus install-all
# (see src/proteus/utils/data.py:get_spider).
#
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Portable realpath: macOS <13 (Catalina through Monterey) does not ship
# GNU coreutils realpath. Fall back to python3, which is always available
# in PROTEUS's conda environment.
# -----------------------------------------------------------------------------
portable_realpath() {
    if command -v realpath >/dev/null 2>&1; then
        realpath "$1"
    else
        python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "$1"
    fi
}

# -----------------------------------------------------------------------------
# Error handling: report which step failed on any non-zero exit
# -----------------------------------------------------------------------------
current_step="initialising"
on_error() {
    local rc=$?  # must be first line — captures the failing command's exit code
    echo ""
    echo "========================================"
    echo " ERROR: SPIDER installation failed"
    echo ""
    echo " Step that failed: $current_step"
    echo " Command:          $BASH_COMMAND"
    echo " Exit code:        $rc"
    echo ""
    echo " Troubleshooting:"
    case "$current_step" in
        *"Cloning"*)
            echo "   - Check your internet connection"
            echo "   - Verify you can reach GitHub: git ls-remote https://github.com/FormingWorlds/SPIDER.git"
            echo "   - If using SSH, check your SSH key: ssh -T git@github.com"
            ;;
        *"Building"*)
            echo "   - Check the compiler output above for errors"
            echo "   - Verify mpicc is working: mpicc --version"
            echo "   - Verify PETSc is intact: ls \$PETSC_DIR/\$PETSC_ARCH/lib/libpetsc.*"
            echo "   - On macOS: ensure SDKROOT is set (xcrun --show-sdk-path)"
            echo "   - See PROTEUS docs/troubleshooting.md for platform-specific fixes"
            ;;
        *"Verif"*)
            echo "   - The build completed without make errors but no binary was produced"
            echo "   - This usually indicates a linker failure that was suppressed"
            echo "   - Try rebuilding with verbose output: cd $workpath && make V=1"
            ;;
        *)
            echo "   - See docs/troubleshooting.md for platform-specific advice"
            ;;
    esac
    echo ""
    echo " PETSc environment used:"
    echo "   PETSC_DIR  = ${PETSC_DIR:-<not set>}"
    echo "   PETSC_ARCH = ${PETSC_ARCH:-<not set>}"
    echo "========================================"
}
trap on_error ERR

# -----------------------------------------------------------------------------
# 1. Detect platform and set PETSC_ARCH
# -----------------------------------------------------------------------------
current_step="Detecting platform"

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PETSC_ARCH=arch-linux-c-opt
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PETSC_ARCH=arch-darwin-c-opt
else
    echo "ERROR: Unsupported OS type '$OSTYPE'. Only Linux and macOS are supported."
    exit 1
fi

# -----------------------------------------------------------------------------
# 2. Locate and validate PETSc installation
# -----------------------------------------------------------------------------
current_step="Validating PETSc installation"

# Derive the repo root from this script's location (tools/get_spider.sh).
# This avoids dependence on the caller's CWD — important when invoked by
# data.py:get_spider() which does not set cwd.
script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(dirname "$script_dir")"

# PETSc is expected at <repo_root>/petsc/.
petsc_path="$repo_root/petsc"
if [[ ! -d "$petsc_path" ]]; then
    echo "ERROR: petsc/ directory not found at $petsc_path."
    echo "Run ./tools/get_petsc.sh first to install PETSc."
    exit 1
fi
PETSC_DIR=$(portable_realpath "$petsc_path")

# Verify PETSc was actually built (not just downloaded/configured).
# The library name varies by platform:
#   macOS:  libpetsc.dylib
#   Linux:  libpetsc.so or libpetsc.so.X.Y (versioned, symlink may be absent)
petsc_lib_dir="$PETSC_DIR/$PETSC_ARCH/lib"
petsc_lib_found=false
for f in "$petsc_lib_dir"/libpetsc.*; do
    if [[ -f "$f" ]]; then
        petsc_lib_found=true
        break
    fi
done
if [[ "$petsc_lib_found" != "true" ]]; then
    echo "ERROR: PETSc library not found in $petsc_lib_dir."
    echo "PETSc may have been downloaded but not compiled successfully."
    echo "Re-run ./tools/get_petsc.sh to rebuild."
    exit 1
fi

# Verify PETSc's Makefile includes exist (required by SPIDER's Makefile)
petsc_conf_dir="$PETSC_DIR/lib/petsc/conf"
if [[ ! -f "$petsc_conf_dir/variables" ]] || \
   [[ ! -f "$petsc_conf_dir/rules" ]]; then
    echo "ERROR: PETSc configuration files not found in $petsc_conf_dir."
    echo "The PETSc installation appears incomplete. Re-run ./tools/get_petsc.sh."
    exit 1
fi

export PETSC_ARCH
export PETSC_DIR

echo "PETSC_DIR  = $PETSC_DIR"
echo "PETSC_ARCH = $PETSC_ARCH"

# -----------------------------------------------------------------------------
# 3. macOS-specific environment setup
# -----------------------------------------------------------------------------
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Set SDKROOT so the compiler can find macOS system headers.
    # Required on Catalina+ where headers are no longer in /usr/include.
    if command -v xcrun >/dev/null 2>&1; then
        export SDKROOT
        SDKROOT=$(xcrun --show-sdk-path)
        echo "SDKROOT    = $SDKROOT"
    fi
fi

# -----------------------------------------------------------------------------
# 4. Verify build tools are available
# -----------------------------------------------------------------------------
current_step="Verifying build tools"

if ! command -v mpicc >/dev/null 2>&1; then
    echo "ERROR: mpicc not found. A C compiler with MPI support is required."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Install via Homebrew: brew install open-mpi"
    else
        echo "Install via package manager, e.g.: sudo apt install libopenmpi-dev"
    fi
    exit 1
fi

if ! command -v make >/dev/null 2>&1; then
    echo "ERROR: make not found. Install build tools for your platform."
    exit 1
fi

if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git not found. Install git to clone the SPIDER repository."
    exit 1
fi

# -----------------------------------------------------------------------------
# 5. Clone SPIDER from GitHub
# -----------------------------------------------------------------------------
current_step="Cloning SPIDER from GitHub"

# Default install directory: ./SPIDER/ ; override via first argument.
workpath="SPIDER"
if [[ -n "$1" ]]; then
    workpath="$1"
fi

# Remove any previous installation
if [[ -d "$workpath" ]]; then
    echo "Removing previous SPIDER installation at $workpath..."
    rm -rf "$workpath"
fi

echo ""
echo "Cloning SPIDER from GitHub..."
git clone https://github.com/FormingWorlds/SPIDER.git "$workpath"

# -----------------------------------------------------------------------------
# 6. Build SPIDER
# -----------------------------------------------------------------------------
current_step="Building SPIDER (make)"

# Determine number of parallel jobs.
# Uses nproc (Linux) or sysctl (macOS) to detect available CPU cores.
if command -v nproc >/dev/null 2>&1; then
    njobs=$(nproc)
elif command -v sysctl >/dev/null 2>&1; then
    njobs=$(sysctl -n hw.ncpu)
else
    njobs=2
fi

echo ""
echo "Building SPIDER ($njobs parallel jobs)..."
olddir=$(pwd)
cd "$workpath"

make -j "$njobs"

# -----------------------------------------------------------------------------
# 7. Verify the build produced the SPIDER binary
# -----------------------------------------------------------------------------
current_step="Verifying SPIDER binary"

if [[ ! -x "spider" ]]; then
    echo "ERROR: SPIDER binary not found after build."
    echo "Check the build output above for compilation errors."
    cd "$olddir"
    exit 1
fi

spider_version=$(./spider --help 2>&1 | head -1 || true)
echo ""
echo "Build successful: $spider_version"

# -----------------------------------------------------------------------------
# 8. Done
# -----------------------------------------------------------------------------
cd "$olddir"

echo ""
echo "========================================"
echo " SPIDER installation complete."
echo ""
echo " Binary: $(portable_realpath "$workpath/spider")"
echo " PETSC_DIR  = $PETSC_DIR"
echo " PETSC_ARCH = $PETSC_ARCH"
echo "========================================"
