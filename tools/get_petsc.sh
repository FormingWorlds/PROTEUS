#!/usr/bin/env bash
# =============================================================================
# get_petsc.sh — Download, configure, and compile PETSc for PROTEUS/SPIDER
# =============================================================================
#
# Downloads PETSc 3.19.0 from OSF and builds it with sundials2 support.
# SPIDER is a pure C code, so C++ and Fortran compilers are disabled.
#
# Supported platforms:
#   - macOS 10.15 (Catalina) and later, Intel and Apple Silicon
#   - Linux (Ubuntu, Debian, Fedora/RHEL, HPC clusters)
#
# Prerequisites:
#   macOS:  brew install gcc open-mpi
#           xcode-select --install
#   Ubuntu: sudo apt install build-essential libopenmpi-dev
#   Fedora: sudo dnf install gcc openmpi openmpi-devel lapack lapack-devel \
#               lapack-static f2c f2c-libs
#
# Usage:
#   ./tools/get_petsc.sh           # install into ./petsc/
#   ./tools/get_petsc.sh /path     # install into /path/petsc/
#
# Environment after completion:
#   PETSC_DIR  = <install path>/petsc
#   PETSC_ARCH = arch-{linux,darwin}-c-opt
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
    echo " ERROR: PETSc installation failed"
    echo ""
    echo " Step that failed: $current_step"
    echo " Command:          $BASH_COMMAND"
    echo " Exit code:        $rc"
    echo ""
    echo " Troubleshooting:"
    case "$current_step" in
        *"Download"*)
            echo "   - Check your internet connection"
            echo "   - Verify the OSF URL is accessible: $url"
            echo "   - Try downloading manually: curl -LsS $url > petsc.zip"
            ;;
        *"Decompress"*)
            echo "   - The downloaded archive may be corrupted"
            echo "   - Delete petsc/ and re-run this script"
            ;;
        *"Configure"*)
            echo "   - Check petsc/configure.log for details"
            echo "   - On macOS: ensure Xcode CLI tools are installed (xcode-select --install)"
            echo "   - Verify MPI is installed (mpicc --version)"
            echo "   - See PROTEUS docs/troubleshooting.md for platform-specific fixes"
            ;;
        *"Build"*)
            echo "   - Check petsc/make.log for compiler errors"
            echo "   - Ensure your C compiler is working (mpicc --version)"
            echo "   - On macOS: verify SDKROOT is set (xcrun --show-sdk-path)"
            ;;
        *"Test"*)
            echo "   - PETSc built but tests failed"
            echo "   - Check petsc/make.log for details"
            echo "   - On macOS: check /etc/hosts for localhost entry"
            echo "     (see docs/troubleshooting.md: PETSc tests error)"
            ;;
        *)
            echo "   - See docs/troubleshooting.md for platform-specific advice"
            ;;
    esac
    echo "========================================"
}
trap on_error ERR

# -----------------------------------------------------------------------------
# 1. Detect platform and set PETSC_ARCH
# -----------------------------------------------------------------------------
current_step="Detecting platform"

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export PETSC_ARCH=arch-linux-c-opt
elif [[ "$OSTYPE" == "darwin"* ]]; then
    export PETSC_ARCH=arch-darwin-c-opt
else
    echo "ERROR: Unsupported OS type '$OSTYPE'. Only Linux and macOS are supported."
    exit 1
fi

# -----------------------------------------------------------------------------
# 2. Set up working directory
# -----------------------------------------------------------------------------
current_step="Setting up working directory"

# Default: ./petsc/ relative to current directory; override via first argument.
# When called from data.py:get_petsc(), the full path is passed as $1.
if [[ -n "$1" ]]; then
    mkdir -p "$1"
    workpath=$(portable_realpath "$1")
else
    mkdir -p petsc
    workpath=$(portable_realpath petsc)
fi

export PETSC_DIR="$workpath"
echo "PETSC_DIR  = $PETSC_DIR"
echo "PETSC_ARCH = $PETSC_ARCH"

# Clean previous installation
rm -rf "$workpath"
mkdir "$workpath"

# -----------------------------------------------------------------------------
# 3. Download PETSc 3.19.0 from OSF
# -----------------------------------------------------------------------------
current_step="Downloading PETSc archive from OSF"

zipfile="$workpath/petsc.zip"
url="https://osf.io/download/p5vxq/"
echo "Downloading PETSc archive from OSF..."
echo "    $url -> $zipfile"
sleep 1
curl -LsS "$url" > "$zipfile"

current_step="Decompressing PETSc archive"
echo "Decompressing..."
unzip -qq "$zipfile" -d "$workpath"
rm "$zipfile"

# -----------------------------------------------------------------------------
# 4. Determine platform-specific configure flags
# -----------------------------------------------------------------------------
current_step="Determining platform-specific flags"

# These variables collect optional flags that vary by platform.
# Defaults assume a generic Linux system without system MPI or BLAS/LAPACK.
mpi_flag="--download-mpich"
blas_flag="--download-f2cblaslapack"
ldflags=""

# ---- Linux special cases ----------------------------------------------------
if [[ "$OSTYPE" == "linux-gnu"* ]]; then

    # Snellius HPC cluster: use the cluster's MPI (loaded via module)
    if [[ "$(hostname -f 2>/dev/null)" == *"snellius"* ]]; then
        echo "    Detected Snellius cluster — using system MPI"
        mpi_flag=""
    fi

    # Fedora / RHEL: system packages provide MPI and BLAS/LAPACK
    if [[ -f "/etc/fedora-release" || -f "/etc/redhat-release" ]]; then
        echo "    Detected Fedora/RHEL — using system MPI and BLAS/LAPACK"
        mpi_flag=""
        blas_flag=""
    fi

    # Generic Linux: if mpicc is available, prefer system MPI over download
    if [[ -n "$mpi_flag" ]] && command -v mpicc >/dev/null 2>&1; then
        echo "    Found system MPI ($(which mpicc)) — skipping mpich download"
        mpi_flag=""
    fi
fi

# ---- macOS -------------------------------------------------------------------
if [[ "$OSTYPE" == "darwin"* ]]; then

    # Verify Xcode Command Line Tools are installed (provides system headers)
    if ! command -v xcrun >/dev/null 2>&1; then
        echo "ERROR: xcrun not found. Install Xcode Command Line Tools:"
        echo "    xcode-select --install"
        exit 1
    fi

    # Set SDKROOT so the compiler can find macOS system headers.
    # Required on Catalina+ where headers are no longer in /usr/include.
    export SDKROOT
    SDKROOT=$(xcrun --show-sdk-path)
    echo "    SDKROOT = $SDKROOT"

    # Use Homebrew's MPI if available (both Intel and Apple Silicon paths)
    if command -v mpicc >/dev/null 2>&1; then
        echo "    Found system MPI ($(which mpicc)) — skipping mpich download"
        mpi_flag=""
    else
        echo "WARNING: mpicc not found. Install MPI via Homebrew:"
        echo "    brew install open-mpi"
        echo "Falling back to --download-mpich (may fail on Apple Silicon)"
    fi

    # macOS provides Accelerate framework with BLAS/LAPACK; no download needed
    blas_flag=""

    # Suppress deprecated linker warnings that break PETSc configure checks.
    # macOS 13+ / Xcode 15+ deprecated -bind_at_load and -multiply_defined;
    # macOS 26+ / clang 17+ treats these warnings as errors in PETSc's
    # configure runtime tests (checkStdC). The -Wl,-w flag suppresses all
    # linker warnings, allowing configure to complete.
    # Homebrew prefix differs by architecture:
    #   Apple Silicon (arm64): /opt/homebrew
    #   Intel (x86_64):        /usr/local
    if [[ "$(uname -m)" == "arm64" ]]; then
        default_brew_prefix="/opt/homebrew"
    else
        default_brew_prefix="/usr/local"
    fi
    brew_prefix=$(brew --prefix 2>/dev/null || echo "$default_brew_prefix")
    ldflags="-L${brew_prefix}/lib -Wl,-w"
fi

# Final check: if we skipped mpich download, mpicc/mpirun must be available
if [[ -z "$mpi_flag" ]] && ! command -v mpirun >/dev/null 2>&1; then
    echo "ERROR: MPI not found and --download-mpich was disabled."
    echo "Install MPI first (e.g. 'brew install open-mpi' or 'apt install libopenmpi-dev')."
    exit 1
fi

# -----------------------------------------------------------------------------
# 5. Configure PETSc
# -----------------------------------------------------------------------------
# Key flags:
#   --with-fc=0   : disable Fortran (SPIDER does not use Fortran)
#   --with-cxx=0  : disable C++ (SPIDER is pure C; also avoids clang 17+
#                    errors in PETSc 3.19's CUDA/CUPM template headers)
#   --download-sundials2 : required by SPIDER for ODE integration
#   --COPTFLAGS   : optimization flags for the C compiler
current_step="Configuring PETSc (./configure)"

echo ""
echo "Configuring PETSc..."
echo "    MPI:  ${mpi_flag:-system}"
echo "    BLAS: ${blas_flag:-system}"
echo "    LDFLAGS: ${ldflags:-<none>}"

olddir=$(pwd)
cd "$workpath"

./configure \
    --with-debugging=0 \
    --with-fc=0 \
    --with-cxx=0 \
    --download-sundials2 \
    --COPTFLAGS="-g -O3" \
    $mpi_flag \
    $blas_flag \
    ${ldflags:+"LDFLAGS=$ldflags"}

# -----------------------------------------------------------------------------
# 6. Build PETSc
# -----------------------------------------------------------------------------
current_step="Building PETSc (make all)"

echo ""
echo "Building PETSc..."
make PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" all

# -----------------------------------------------------------------------------
# 7. Run PETSc self-tests
# -----------------------------------------------------------------------------
current_step="Testing PETSc (make check)"

echo ""
echo "Testing PETSc..."
make PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" check

# -----------------------------------------------------------------------------
# 8. Done
# -----------------------------------------------------------------------------
cd "$olddir"

echo ""
echo "========================================"
echo " PETSc installation complete."
echo ""
echo " PETSC_DIR  = $PETSC_DIR"
echo " PETSC_ARCH = $PETSC_ARCH"
echo ""
echo " Add these to your shell config if you"
echo " need to rebuild SPIDER manually:"
echo "   export PETSC_DIR=$PETSC_DIR"
echo "   export PETSC_ARCH=$PETSC_ARCH"
echo "========================================"
