#!/usr/bin/env bash
# PROTEUS unified installer
#
# Usage:
#   bash install.sh              # Essential install (spectral + stellar data)
#   bash install.sh --all-data   # Full install (all reference data)
#   bash install.sh --no-data    # Skip data downloads
#   bash install.sh -i           # Interactive mode (prompt for choices)
#   bash install.sh --help       # Show usage
#
# Prerequisites:
#   - conda environment with Python 3.12 must be active
#   - System packages (gfortran >= 9, netcdf) must be installed
#   - Internet connection for downloads
#
# This script is idempotent: safe to re-run after a failure or update.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Python target: fallback values, overridden from pyproject.toml's
# requires-python ceiling once the repo root is known (see below).
REQUIRED_PYTHON_MAJOR=3
REQUIRED_PYTHON_MINOR=12
# Julia: 1.11.x and 1.12.x are both supported; fresh installs pin the
# version below (matches what CI tests).
REQUIRED_JULIA_MAJOR=1
REQUIRED_JULIA_MINOR=12
ACCEPTED_JULIA_MINORS="11 12"
MIN_DISK_GB=10

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { printf "${GREEN}[INFO]${NC}  %s\n" "$*"; }
warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
fail()  { printf "${RED}[FAIL]${NC}  %s\n" "$*"; }
phase() { printf "\n${CYAN}${BOLD}=== Phase %s: %s ===${NC}\n" "$1" "$2"; }

die() {
    fail "$1"
    echo ""
    echo "Installation failed at Phase $CURRENT_PHASE."
    if [ -n "${LOGFILE:-}" ] && [ -f "${LOGFILE:-}" ]; then
        echo "See $LOGFILE for details."
    fi
    echo "Fix the issue above, then re-run: bash install.sh"
    exit 1
}

prompt_yn() {
    local msg="$1" default="${2:-y}"
    if [ "$INTERACTIVE" != "true" ] || [ ! -t 0 ]; then
        return 0
    fi
    if [ "$default" = "y" ]; then
        read -r -p "$msg [Y/n] " response
        response="${response:-y}"
    else
        read -r -p "$msg [y/N] " response
        response="${response:-n}"
    fi
    [[ "$response" =~ ^[Yy] ]]
}

read_with_default() {
    local prompt="$1" default="$2"
    if [ "$INTERACTIVE" != "true" ] || [ ! -t 0 ]; then
        echo "$default"
        return
    fi
    read -r -p "$prompt" response
    echo "${response:-$default}"
}

command_exists() { command -v "$1" &>/dev/null; }

exit_export_message=""

print_exit_export_message() {
    if [ -n "$exit_export_message" ]; then
        echo ""
        echo "--- NOTE: Run the following export commands manually before restarting this install script if you want it to skip steps that were successful (or source your shell rc file now if it was modified):"
        printf "%s" "$exit_export_message"
    fi
}

trap print_exit_export_message EXIT

# Detect the shell rc file for the current user
detect_shell_rc() {
    local shell_name
    shell_name="$(basename "${SHELL:-/bin/bash}")"
    case "$shell_name" in
        zsh)  echo "$HOME/.zshrc" ;;
        bash) echo "$HOME/.bashrc" ;;
        *)
            warn "Unsupported shell '$shell_name' for auto-configuration; writing to ~/.bashrc"
            echo "$HOME/.bashrc"
            ;;
    esac
}

# Safely append an export line to the shell rc file
append_export_to_rc() {
    local var_name="$1" var_value="$2" rc_file="$3"
    local safe_value
    safe_value=$(printf '%q' "$var_value")
    local line="export ${var_name}=${safe_value}"
    # Remove any existing line for this variable before appending
    if [ -f "$rc_file" ]; then
        grep -v "^export ${var_name}=" "$rc_file" > "${rc_file}.tmp" 2>/dev/null || true
        mv "${rc_file}.tmp" "$rc_file"
    fi
    echo "$line" >> "$rc_file"
    exit_export_message+="$line"
    exit_export_message+=$'\n'
    info "Set in $rc_file: export ${var_name}=${var_value}"
}

# Get available disk space in GB (POSIX-portable)
available_disk_gb() {
    df -k . | awk 'NR==2 {printf "%d\n", $4/1024/1024}'
}

CURRENT_PHASE=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERACTIVE="false"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

DATA_MODE="essential"
for arg in "$@"; do
    case "$arg" in
        --all-data) DATA_MODE="all" ;;
        --no-data)  DATA_MODE="none" ;;
        --interactive|-i) INTERACTIVE="true" ;;
        --help|-h)
            echo "PROTEUS unified installer"
            echo ""
            echo "Usage: bash install.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --all-data   Download all reference data (~10-20 GB)"
            echo "  --no-data    Skip data downloads entirely"
            echo "  -i, --interactive  Interactive mode (prompt for choices)"
            echo "  --help       Show this message"
            echo ""
            echo "Default: download essential data (~2 GB)"
            exit 0
            ;;
        *)
            fail "Unknown argument: $arg"
            echo "Run 'bash install.sh --help' for usage."
            exit 1
            ;;
    esac
done

# Start logging
LOGFILE="$SCRIPT_DIR/install_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

echo ""
printf "${BOLD}PROTEUS Installer${NC}\n"
echo "Log: $LOGFILE"
echo "Data mode: $DATA_MODE"
if [ "$INTERACTIVE" = "true" ]; then
    echo "Mode: interactive (-i)"
fi
echo ""

# ===================================================================
# Phase 1: Pre-flight checks
# ===================================================================
CURRENT_PHASE=1
phase 1 "Pre-flight checks"

# Detect OS
OS="unknown"
ARCH="$(uname -m)"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    info "Detected macOS ($ARCH)"
elif [[ "$OSTYPE" == "linux"* ]]; then
    # WSL2 detection
    if uname -r 2>/dev/null | grep -qi microsoft; then
        warn "WSL2 detected. PROTEUS is tested on native Linux; WSL2 may have quirks."
    fi
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        case "$ID" in
            ubuntu|debian|pop|linuxmint) OS="debian" ;;
            fedora|rhel|centos|rocky|alma) OS="fedora" ;;
            alpine) OS="alpine" ;;
            *) OS="linux-other" ;;
        esac
        info "Detected Linux: $PRETTY_NAME ($ARCH)"
    else
        OS="linux-other"
        info "Detected Linux ($ARCH)"
    fi
else
    die "Unsupported OS: $OSTYPE. PROTEUS requires macOS or Linux."
fi

# Check we are in the PROTEUS repo
if [ ! -f "$SCRIPT_DIR/pyproject.toml" ]; then
    die "install.sh must be run from the PROTEUS repository root."
fi
cd "$SCRIPT_DIR"

# Derive the Python target from pyproject.toml's requires-python ceiling
# (e.g. ">=3.11,<3.13" targets 3.12), so the installer and the package
# metadata cannot drift apart. The fallback above applies when the
# ceiling cannot be parsed.
python_ceiling=$(grep -m1 'requires-python' pyproject.toml \
    | grep -oE '<3\.[0-9]+' | grep -oE '[0-9]+$' || true)
if [ -n "$python_ceiling" ]; then
    REQUIRED_PYTHON_MINOR=$((python_ceiling - 1))
else
    warn "Could not parse the requires-python ceiling from pyproject.toml;"
    warn "falling back to Python $REQUIRED_PYTHON_MAJOR.$REQUIRED_PYTHON_MINOR."
fi

# Check disk space
disk_gb=$(available_disk_gb)
if [ "$disk_gb" -lt "$MIN_DISK_GB" ]; then
    die "Insufficient disk space: ${disk_gb} GB available, ${MIN_DISK_GB} GB required."
fi
info "Disk space: ${disk_gb} GB available"

# Check conda environment is active
if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    fail "No conda environment is active."
    echo ""
    echo "Create and activate a conda environment first:"
    echo "  conda create -n proteus python=$REQUIRED_PYTHON_MAJOR.$REQUIRED_PYTHON_MINOR"
    echo "  conda activate proteus"
    die "Conda environment required."
fi
info "Conda environment: $CONDA_DEFAULT_ENV"

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "none")
if [ "$python_version" = "none" ]; then
    die "Python 3 not found in PATH."
fi
py_major=$(echo "$python_version" | cut -d. -f1)
py_minor=$(echo "$python_version" | cut -d. -f2)
if [ "$py_major" -ne "$REQUIRED_PYTHON_MAJOR" ] || [ "$py_minor" -ne "$REQUIRED_PYTHON_MINOR" ]; then
    die "Python $REQUIRED_PYTHON_MAJOR.$REQUIRED_PYTHON_MINOR required, found $python_version."
fi
info "Python: $python_version ($(which python3))"

# Check system dependencies
missing_deps=()

if ! command_exists gfortran; then
    missing_deps+=("gfortran")
else
    gf_version=$(gfortran -dumpversion 2>/dev/null | cut -d. -f1)
    if [ -n "$gf_version" ] && [ "$gf_version" -lt 9 ] 2>/dev/null; then
        warn "gfortran $gf_version detected; SOCRATES requires >= 9. You may see build failures."
    fi
fi
if ! command_exists git; then
    missing_deps+=("git")
fi
if ! command_exists make; then
    missing_deps+=("make")
fi
if ! command_exists curl; then
    missing_deps+=("curl")
fi
if ! command_exists cmake; then
    missing_deps+=("cmake")
fi
if ! command_exists unzip; then
    missing_deps+=("unzip")
fi

# NetCDF (needed for SOCRATES)
if ! command_exists nc-config && ! command_exists nf-config; then
    missing_deps+=("netcdf-dev")
fi

if [ ${#missing_deps[@]} -gt 0 ]; then
    fail "Missing system packages: ${missing_deps[*]}"
    echo ""
    echo "Install them with:"
    case "$OS" in
        macos)
            echo "  brew install gcc netcdf netcdf-fortran wget cmake"
            ;;
        debian)
            echo "  sudo apt install gfortran libnetcdff-dev build-essential curl git cmake unzip"
            ;;
        fedora)
            echo "  sudo dnf install gcc-gfortran netcdf-fortran-devel make curl git cmake unzip"
            ;;
        alpine)
            echo "  apk add gfortran netcdf-fortran-dev make curl git cmake unzip"
            ;;
        *)
            echo "  Install: ${missing_deps[*]}"
            echo "  (package names vary by distribution)"
            ;;
    esac
    echo ""
    echo "Then re-run: bash install.sh"
    die "Missing system dependencies."
fi
info "System dependencies: OK"

# HPC / NFS-home detection. On clusters like Kapteyn the home directory is
# small (< 10 GB) and pip/Julia caches must be redirected to a data volume.
HOME_SMALL=false
home_gb=$(df -k "$HOME" | awk 'NR==2 {printf "%d\n", $2/1024/1024}')
if [ "$home_gb" -lt 20 ] 2>/dev/null; then
    HOME_SMALL=true
    warn "Small home directory detected (${home_gb} GB total)."
    warn "pip and Julia caches may exceed this quota."
    # Redirect pip cache if not already set
    if [ -z "${PIP_CACHE_DIR:-}" ]; then
        pip_cache="$SCRIPT_DIR/.pip-cache"
        mkdir -p "$pip_cache"
        export PIP_CACHE_DIR="$pip_cache"
        info "Set PIP_CACHE_DIR=$pip_cache (avoids filling home quota)"
    fi
fi

info "Pre-flight checks passed"

# ===================================================================
# Phase 2: Julia
# ===================================================================
CURRENT_PHASE=2
phase 2 "Julia"

if command_exists julia; then
    julia_version=$(julia --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
    if [ -z "$julia_version" ]; then
        die "Could not parse Julia version from 'julia --version' output."
    fi
    julia_major=$(echo "$julia_version" | cut -d. -f1)
    julia_minor=$(echo "$julia_version" | cut -d. -f2)
    info "Julia found: $julia_version"

    julia_minor_ok=false
    for accepted_minor in $ACCEPTED_JULIA_MINORS; do
        if [ "$julia_major" -eq "$REQUIRED_JULIA_MAJOR" ] && \
           [ "$julia_minor" -eq "$accepted_minor" ]; then
            julia_minor_ok=true
        fi
    done
    if [ "$julia_minor_ok" != true ]; then
        warn "Julia $REQUIRED_JULIA_MAJOR.x with minor in {$ACCEPTED_JULIA_MINORS} required, found $julia_version."
        if command_exists juliaup; then
            info "Pinning Julia to $REQUIRED_JULIA_MAJOR.$REQUIRED_JULIA_MINOR via juliaup..."
            juliaup add "$REQUIRED_JULIA_MAJOR.$REQUIRED_JULIA_MINOR"
            juliaup default "$REQUIRED_JULIA_MAJOR.$REQUIRED_JULIA_MINOR"
            info "Julia pinned to $REQUIRED_JULIA_MAJOR.$REQUIRED_JULIA_MINOR"
        else
            warn "juliaup not found. Cannot pin Julia version automatically."
            echo "  Install juliaup: curl -fsSL https://install.julialang.org | sh"
            echo "  Then: juliaup add $REQUIRED_JULIA_MAJOR.$REQUIRED_JULIA_MINOR"
            echo "        juliaup default $REQUIRED_JULIA_MAJOR.$REQUIRED_JULIA_MINOR"
            die "Julia version mismatch."
        fi
    fi
else
    info "Julia not found."
    if prompt_yn "Install Julia via juliaup (official installer)?"; then
        info "Installing Julia..."
        # Download to temp file for auditability
        julia_installer=$(mktemp /tmp/juliaup-install-XXXXXX.sh)
        curl -fsSL https://install.julialang.org -o "$julia_installer"
        bash "$julia_installer" --yes 2>&1
        rm -f "$julia_installer"
        # Source the juliaup env for the current shell
        if [ -f "$HOME/.juliaup/bin/juliaup" ]; then
            export PATH="$HOME/.juliaup/bin:$PATH"
        fi
        if ! command_exists juliaup; then
            die "juliaup installation failed. Install Julia manually: https://julialang.org/downloads/"
        fi
        juliaup add "$REQUIRED_JULIA_MAJOR.$REQUIRED_JULIA_MINOR"
        juliaup default "$REQUIRED_JULIA_MAJOR.$REQUIRED_JULIA_MINOR"
        info "Julia $REQUIRED_JULIA_MAJOR.$REQUIRED_JULIA_MINOR installed"
    else
        die "Julia is required for AGNI. Install it manually: https://julialang.org/downloads/"
    fi
fi

# Verify Julia works
if ! julia -e 'println("ok")' &>/dev/null; then
    die "Julia is installed but not functional. Check your PATH."
fi
info "Julia: OK"

# ===================================================================
# Phase 3: Environment variables
# ===================================================================
CURRENT_PHASE=3
phase 3 "Environment variables"

RC_FILE=$(detect_shell_rc)
# Ensure parent directory exists (e.g. ~/.config/fish/)
mkdir -p "$(dirname "$RC_FILE")"
info "Shell config: $RC_FILE"

# FWL_DATA
if [ -z "${FWL_DATA:-}" ]; then
    default_fwl="$HOME/FWL_DATA"
    info "FWL_DATA is not set."
    fwl_path=$(read_with_default "Data directory [$default_fwl]: " "$default_fwl")
    export FWL_DATA="$fwl_path"
else
    fwl_path="$FWL_DATA"
    info "FWL_DATA already set: $FWL_DATA"
fi
mkdir -p "$fwl_path"
append_export_to_rc "FWL_DATA" "$fwl_path" "$RC_FILE"

# PYTHON_JULIAPKG_EXE
julia_exe="$(which julia)"
export PYTHON_JULIAPKG_EXE="$julia_exe"
append_export_to_rc "PYTHON_JULIAPKG_EXE" "$julia_exe" "$RC_FILE"

info "Environment variables configured"

# ===================================================================
# Phase 4: SOCRATES (Fortran radiative transfer)
# ===================================================================
CURRENT_PHASE=4
phase 4 "SOCRATES"

socrates_compiled=false
if [ -n "${RAD_DIR:-}" ] && [ -d "${RAD_DIR}" ]; then
    if [ -f "${RAD_DIR}/bin/radlib.a" ]; then
        socrates_compiled=true
    fi
fi

if [ "$socrates_compiled" = "true" ]; then
    info "SOCRATES already installed at $RAD_DIR"
else
    info "Installing SOCRATES..."
    # On HPC clusters, conda's netcdf can shadow the system netcdf and
    # cause SOCRATES to link against the wrong libraries. Temporarily
    # hide the conda lib/ from the linker so gfortran finds the system
    # netcdf-fortran. The conda env stays active for Python; only the
    # Fortran linker path is adjusted.
    _saved_conda_prefix="${CONDA_PREFIX:-}"
    if [ -n "$_saved_conda_prefix" ] && [ -d "$_saved_conda_prefix/lib" ]; then
        _saved_ld="${LD_LIBRARY_PATH:-}"
        export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v "$_saved_conda_prefix" | tr '\n' ':' | sed 's/:$//')
        _saved_library="${LIBRARY_PATH:-}"
        export LIBRARY_PATH=$(echo "${LIBRARY_PATH:-}" | tr ':' '\n' | grep -v "$_saved_conda_prefix" | tr '\n' ':' | sed 's/:$//')
        info "Temporarily hiding conda lib paths from Fortran linker"
    fi
    bash tools/get_socrates.sh
    # Restore conda paths
    if [ -n "$_saved_conda_prefix" ] && [ -d "$_saved_conda_prefix/lib" ]; then
        export LD_LIBRARY_PATH="$_saved_ld"
        export LIBRARY_PATH="$_saved_library"
    fi
    # Verify compilation succeeded (radlib.a is the primary build artifact)
    if [ -d "$SCRIPT_DIR/socrates" ] && [ -f "$SCRIPT_DIR/socrates/bin/radlib.a" ]; then
        export RAD_DIR="$SCRIPT_DIR/socrates"
        append_export_to_rc "RAD_DIR" "$SCRIPT_DIR/socrates" "$RC_FILE"
        info "SOCRATES installed, RAD_DIR=$RAD_DIR"
    else
        die "SOCRATES compilation failed. Check $LOGFILE for build errors."
    fi
fi

# ===================================================================
# Phase 5: AGNI + FastChem (Julia atmosphere model + equilibrium chemistry)
# ===================================================================
CURRENT_PHASE=5
phase 5 "AGNI + FastChem"

if [ -d "$SCRIPT_DIR/AGNI" ] && [ -f "$SCRIPT_DIR/AGNI/Manifest.toml" ]; then
    info "AGNI already installed at $SCRIPT_DIR/AGNI"
else
    info "Installing AGNI..."
    bash tools/get_agni.sh 0 2>&1
    if [ ! -f "$SCRIPT_DIR/AGNI/Manifest.toml" ]; then
        die "AGNI installation failed (Julia packages not resolved). Check $LOGFILE for details."
    fi
    info "AGNI installed"
fi

# FastChem (equilibrium chemistry solver used by AGNI)
# Resolve the real AGNI path: on clusters AGNI may be a symlink to NFS,
# and git operations on NFS can fail with index.lock errors. We resolve
# the symlink so paths are canonical and NFS locking is less likely to
# trip on path mismatches.
AGNI_REAL=$(cd "$SCRIPT_DIR/AGNI" 2>/dev/null && pwd -P)
if [ -d "$AGNI_REAL/fastchem" ] && [ -f "$AGNI_REAL/fastchem/fastchem" ]; then
    info "FastChem already installed at $AGNI_REAL/fastchem"
else
    info "Installing FastChem..."
    cd "$AGNI_REAL"
    # If a previous NFS-failed clone left a partial fastchem dir, clean it
    if [ -d "$AGNI_REAL/fastchem/.git" ] && [ ! -f "$AGNI_REAL/fastchem/CMakeLists.txt" ]; then
        warn "Cleaning up partial FastChem clone from a previous failed attempt"
        rm -rf "$AGNI_REAL/fastchem"
    fi
    bash src/get_fastchem.sh -y 2>&1
    cd "$SCRIPT_DIR"
    if [ ! -d "$AGNI_REAL/fastchem" ]; then
        warn "FastChem installation failed. AGNI chemistry features will not work."
        warn "Install manually: cd AGNI && bash src/get_fastchem.sh"
    else
        info "FastChem installed"
    fi
fi

# FC_DIR (use resolved AGNI path so the env var survives symlink changes)
#   note: get_fastchem.sh also sets FC_DIR; we keep it here for consistency and for the rerun environment variable export message
if [ -d "$AGNI_REAL/fastchem" ]; then
    export FC_DIR="$AGNI_REAL/fastchem"
    append_export_to_rc "FC_DIR" "$AGNI_REAL/fastchem" "$RC_FILE"
fi

# ===================================================================
# Phase 6: Python submodules (editable installs)
# ===================================================================
CURRENT_PHASE=6
phase 6 "Python submodules"

# Detect SSH access to GitHub (exit code 1 = authenticated, other = no SSH)
if ssh -T git@github.com; then
    GH_PREFIX="https://github.com/"
    info "GitHub SSH access: not available, using HTTPS"
else
    if [ $? -eq 1 ]; then
        GH_PREFIX="git@github.com:"
        info "GitHub SSH access: OK"
    else
        die "Unexpected SSH exit code $? when testing GitHub access."
    fi
fi

# Clone and install a submodule as editable if not already present.
# Optional 4th arg: branch to checkout (defaults to repo default branch).
clone_and_install() {
    local name="$1" org="$2" repo="$3" branch="${4:-}"
    local dest="$SCRIPT_DIR/$repo"
    if [ -d "$dest" ]; then
        info "$name already cloned at $dest"
    else
        info "Cloning $name..."
        if [ -n "$branch" ]; then
            git clone -b "$branch" "${GH_PREFIX}${org}/${repo}.git" "$dest"
        else
            git clone "${GH_PREFIX}${org}/${repo}.git" "$dest"
        fi
    fi
    info "Installing $name (editable)..."
    pip install -e "$dest/." 2>&1
}

clone_and_install "MORS"     "FormingWorlds" "MORS"
clone_and_install "JANUS"    "FormingWorlds" "JANUS"
clone_and_install "CALLIOPE" "FormingWorlds" "CALLIOPE"
clone_and_install "ZEPHYRUS" "FormingWorlds" "ZEPHYRUS"

# Aragog and Zalmoxis use dedicated setup scripts
info "Setting up Aragog..."
bash tools/get_aragog.sh 2>&1
info "Setting up Zalmoxis..."
bash tools/get_zalmoxis.sh 2>&1

# Install PROTEUS itself
info "Installing PROTEUS and remaining dependencies..."
pip install -e ".[develop]"

info "Setting up pre-commit hooks..."
pre-commit install -f 2>&1 || warn "pre-commit install failed (non-critical)"

# Verify the installation
if ! python3 -c "import proteus" 2>/dev/null; then
    die "PROTEUS Python package failed to import. Check pip install output in $LOGFILE."
fi
info "Python packages installed"

# ===================================================================
# Phase 7: Reference data
# ===================================================================
CURRENT_PHASE=7
phase 7 "Reference data"

case "$DATA_MODE" in
    essential)
        info "Downloading essential reference data (~2 GB)..."
        # Default-config and tutorial k-tables only; the bare
        # `proteus get spectral` downloads every group (~10 GB).
        # Honeyside/48 serves the config default (src/proteus/config/_atmos_clim.py),
        # Dayspring/48 serves input/tutorials/*.toml; update here when those change.
        proteus get spectral -n Honeyside -b 48 2>&1 || warn "Spectral data download had issues"
        proteus get spectral -n Dayspring -b 48 2>&1 || warn "Spectral data download had issues"
        proteus get stellar 2>&1 || warn "Stellar data download had issues"
        # Interior lookup tables, melting curves, and the structure-solver
        # EOS tables that the default and tutorial configs require.
        proteus get interiordata 2>&1 || warn "Interior data download had issues"
        info "Essential data downloaded"
        ;;
    all)
        info "Downloading all reference data (~10-20 GB, this may take a while)..."
        proteus get spectral 2>&1 || warn "Spectral data download had issues"
        proteus get stellar 2>&1 || warn "Stellar data download had issues"
        proteus get muscles --all 2>&1 || warn "MUSCLES download had issues"
        proteus get phoenix --feh 0.0 --alpha 0.0 2>&1 || warn "PHOENIX download had issues"
        proteus get reference 2>&1 || warn "Reference data download had issues"
        proteus get interiordata 2>&1 || warn "Interior data download had issues"
        info "All data downloaded"
        ;;
    none)
        info "Skipping data downloads (use 'proteus get' to download later)"
        ;;
esac

# ===================================================================
# Phase 8: Verification
# ===================================================================
CURRENT_PHASE=8
phase 8 "Verification"

info "Running proteus doctor..."
echo ""
proteus doctor 2>&1 || true
echo ""

# ===================================================================
# Done
# ===================================================================
echo ""
printf "${GREEN}${BOLD}Installation complete!${NC}\n"
echo ""
echo "Next steps:"
echo "  1. Source your shell config:  source $RC_FILE"
echo "  2. Run the quick-start tutorial:"
echo "     proteus start --offline -c input/dummy.toml"
echo ""
echo "For SPIDER (legacy interior module), run separately:"
echo "  bash tools/get_petsc.sh && bash tools/get_spider.sh"
echo ""
echo "Full documentation: https://proteus-framework.org/PROTEUS/"
echo "Log saved to: $LOGFILE"
