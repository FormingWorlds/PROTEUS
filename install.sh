#!/usr/bin/env bash
# PROTEUS unified installer
#
# Usage:
#   bash install.sh              # Essential install (spectral + stellar data)
#   bash install.sh --all-data   # Full install (all reference data)
#   bash install.sh --no-data    # Skip data downloads
#   bash install.sh --yes        # Non-interactive mode (accept all defaults)
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

REQUIRED_PYTHON_MAJOR=3
REQUIRED_PYTHON_MINOR=12
REQUIRED_JULIA_MAJOR=1
REQUIRED_JULIA_MINOR=11
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
    if [ "$NON_INTERACTIVE" = "true" ]; then
        return 0
    fi
    if [ ! -t 0 ]; then
        warn "Non-interactive shell detected, assuming yes for: $msg"
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
    if [ "$NON_INTERACTIVE" = "true" ] || [ ! -t 0 ]; then
        echo "$default"
        return
    fi
    read -r -p "$prompt" response
    echo "${response:-$default}"
}

command_exists() { command -v "$1" &>/dev/null; }

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
    info "Set in $rc_file: export ${var_name}=${var_value}"
}

# Get available disk space in GB (POSIX-portable)
available_disk_gb() {
    df -k . | awk 'NR==2 {printf "%d\n", $4/1024/1024}'
}

CURRENT_PHASE=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NON_INTERACTIVE="false"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

DATA_MODE="essential"
for arg in "$@"; do
    case "$arg" in
        --all-data) DATA_MODE="all" ;;
        --no-data)  DATA_MODE="none" ;;
        --yes|-y)   NON_INTERACTIVE="true" ;;
        --help|-h)
            echo "PROTEUS unified installer"
            echo ""
            echo "Usage: bash install.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --all-data   Download all reference data (~10-20 GB)"
            echo "  --no-data    Skip data downloads entirely"
            echo "  --yes, -y    Non-interactive mode (accept all defaults)"
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
if [ "$NON_INTERACTIVE" = "true" ]; then
    echo "Mode: non-interactive (--yes)"
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
    echo "  conda create -n proteus python=3.12"
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
            echo "  brew install gcc netcdf netcdf-fortran wget"
            ;;
        debian)
            echo "  sudo apt install gfortran libnetcdff-dev build-essential curl git"
            ;;
        fedora)
            echo "  sudo dnf install gcc-gfortran netcdf-fortran-devel make curl git"
            ;;
        alpine)
            echo "  apk add gfortran netcdf-fortran-dev make curl git"
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

    if [ "$julia_major" -ne "$REQUIRED_JULIA_MAJOR" ] || [ "$julia_minor" -ne "$REQUIRED_JULIA_MINOR" ]; then
        warn "Julia $REQUIRED_JULIA_MAJOR.$REQUIRED_JULIA_MINOR required, found $julia_version."
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
    # Check for compiled binaries (radlib.a or Cl_run)
    if [ -f "${RAD_DIR}/bin/Cl_run" ] || find "${RAD_DIR}" -name 'radlib.a' -print -quit 2>/dev/null | grep -q .; then
        socrates_compiled=true
    fi
fi

if [ "$socrates_compiled" = "true" ]; then
    info "SOCRATES already installed at $RAD_DIR"
else
    info "Installing SOCRATES..."
    bash tools/get_socrates.sh
    # Verify compilation succeeded
    if [ -d "$SCRIPT_DIR/socrates" ] && [ -f "$SCRIPT_DIR/socrates/bin/Cl_run" ]; then
        export RAD_DIR="$SCRIPT_DIR/socrates"
        append_export_to_rc "RAD_DIR" "$SCRIPT_DIR/socrates" "$RC_FILE"
        info "SOCRATES installed, RAD_DIR=$RAD_DIR"
    else
        die "SOCRATES compilation failed. Check $LOGFILE for build errors."
    fi
fi

# ===================================================================
# Phase 5: AGNI (Julia atmosphere model)
# ===================================================================
CURRENT_PHASE=5
phase 5 "AGNI"

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

# ===================================================================
# Phase 6: Python packages
# ===================================================================
CURRENT_PHASE=6
phase 6 "Python packages"

info "Installing PROTEUS and dependencies..."
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
        proteus get spectral 2>&1 || warn "Spectral data download had issues"
        proteus get stellar 2>&1 || warn "Stellar data download had issues"
        info "Essential data downloaded"
        ;;
    all)
        info "Downloading all reference data (~10-20 GB, this may take a while)..."
        proteus get spectral 2>&1 || warn "Spectral data download had issues"
        proteus get stellar 2>&1 || warn "Stellar data download had issues"
        proteus get muscles --all 2>&1 || warn "MUSCLES download had issues"
        proteus get phoenix 2>&1 || warn "PHOENIX download had issues"
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
