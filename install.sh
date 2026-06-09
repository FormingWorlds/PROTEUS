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

# Collect machine and environment details into the log so the log file alone is
# enough for us to diagnose an install failure, without a back-and-forth.
# This runs from die() under `set -euo pipefail`, so every command here must be
# safe against errexit: a grep that matches nothing must not abort the script.
collect_env_info() {
    echo ""
    echo "=== Environment (auto-collected for debugging) ==="
    echo "date:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "uname:   $(uname -a)"
    command_exists sw_vers && echo "macos:   $(sw_vers -productVersion 2>/dev/null) ($(uname -m))"
    echo "shell:   ${SHELL:-?}"
    echo "conda:   env=${CONDA_DEFAULT_ENV:-(none)} prefix=${CONDA_PREFIX:-?}"
    echo "python:  $(python3 --version 2>&1) ($(command -v python3 2>/dev/null))"
    echo "julia:   $(julia --version 2>/dev/null || echo '(not on PATH)') ($(command -v julia 2>/dev/null))"
    local v
    for v in FWL_DATA RAD_DIR FC_DIR PYTHON_JULIAPKG_EXE PYTHON_JULIACALL_BINDIR; do
        echo "  ${v}=${!v:-(unset)}"
    done
    echo "package versions:"
    python3 -m pip list 2>/dev/null | grep -iE 'fwl-|juliacall|juliapkg|^jax |jaxlib|equinox|netcdf|hdf5' | sed 's/^/  /' || true
    if [ -n "${SCRIPT_DIR:-}" ] && [ -d "${SCRIPT_DIR}/.git" ]; then
        echo "proteus git: $(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null) ($(git -C "$SCRIPT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null))"
    fi
    echo "conda HDF5/netCDF/MPI builds:"
    { command_exists conda && conda list 2>/dev/null | grep -iE '^(hdf5|libnetcdf|netcdf4|mpich|openmpi|libmpi)\b' | sed 's/^/  /'; } || true
    echo "=== end environment ==="
}

die() {
    fail "$1"
    echo ""
    echo "Installation failed at Phase $CURRENT_PHASE."
    collect_env_info || true
    echo ""
    if [ -n "${LOGFILE:-}" ] && [ -f "${LOGFILE:-}" ]; then
        echo "A full log was written to: $LOGFILE"
        echo "If you need help, send that log file to proteus_dev@formingworlds.space,"
        echo "or open an issue or discussion at:"
        echo "  https://github.com/FormingWorlds/PROTEUS/issues"
        echo "  https://github.com/orgs/FormingWorlds/discussions"
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

# A fresh conda environment (notably anaconda3 on macOS) often installs the
# MPI-enabled builds of HDF5/libnetcdf. Those load libmpi before Julia, and the
# duplicate MPI symbols crash the juliacall bridge when "import proteus" loads
# it. PROTEUS uses no MPI in its Python stack, so the no-MPI builds are correct.
conda_has_mpi_netcdf() {
    command_exists conda || return 1
    # MPI builds carry a "mpi_..." build string; no-MPI builds carry "nompi_...".
    conda list 2>/dev/null \
        | awk '($1=="hdf5"||$1=="libnetcdf"){print $1, $3}' \
        | grep -vi 'nompi' | grep -qi 'mpi'
}

fix_conda_mpi_builds() {
    command_exists conda || { warn "conda not on PATH; cannot adjust HDF5/netCDF builds."; return 1; }
    info "Installing the no-MPI builds of HDF5 and libnetcdf..."
    conda install -y -c conda-forge "hdf5=*=nompi*" "libnetcdf=*=nompi*" 2>&1 || return 1
}

# juliaup installs a launcher shim at ~/.juliaup/bin/julia that dispatches to
# the real versioned binary. juliacall needs the real binary as
# PYTHON_JULIAPKG_EXE so it can find sys.so / libjulia beside it (the shim has
# no lib/julia/sys.so). Ask the running Julia for its actual BINDIR; fall back
# to the launcher only if that probe fails.
resolve_real_julia_exe() {
    local bindir
    bindir="$(julia --startup-file=no -e 'print(Sys.BINDIR)' 2>/dev/null || true)"
    if [ -n "$bindir" ] && [ -x "$bindir/julia" ]; then
        printf '%s\n' "$bindir/julia"
    else
        command -v julia 2>/dev/null || true
    fi
}

# juliacall builds a Julia environment whose OpenSSL_jll is matched to the
# OpenSSL the Python interpreter links against. Julia 1.12 provides OpenSSL_jll
# 3.5 and newer only, so a Python interpreter linking OpenSSL < 3.5 pins
# OpenSSL_jll to the 3.0 series and leaves the Julia 1.12 resolve unsatisfiable.
python_openssl_below_35() {
    python3 - <<'PY' 2>/dev/null
import ssl, sys
sys.exit(0 if ssl.OPENSSL_VERSION_INFO[:2] < (3, 5) else 1)
PY
}

julia_minor_at_least_12() {
    local v maj min
    v=$(julia --version 2>/dev/null | awk '{print $3}')
    maj=${v%%.*}
    min=$(printf '%s' "$v" | cut -d. -f2)
    case "$maj.$min" in *[!0-9.]*|.|"") return 1 ;; esac
    [ "$maj" -eq 1 ] && [ "$min" -ge 12 ]
}

conda_openssl_too_old_for_julia() {
    command_exists conda || return 1
    julia_minor_at_least_12 || return 1
    python_openssl_below_35 || return 1
    return 0
}

fix_conda_openssl() {
    command_exists conda || { warn "conda not on PATH; cannot upgrade OpenSSL."; return 1; }
    info "Upgrading OpenSSL to >= 3.5 (required by the Julia 1.12 environment)..."
    # Redirect stdin so an unexpected channel Terms-of-Service prompt fails fast
    # instead of hanging a non-interactive install.
    conda install -y -c conda-forge "openssl>=3.5" </dev/null 2>&1 || return 1
    # The upgrade only helps if the interpreter actually links the new OpenSSL;
    # a defaults-channel env can report success while leaving the link unchanged.
    if python_openssl_below_35; then
        warn "conda reported success but Python still links OpenSSL < 3.5:"
        warn "  $(python3 -c 'import ssl; print(ssl.OPENSSL_VERSION)' 2>/dev/null)"
        return 1
    fi
    # Drop the installer-owned juliacall project so the next import re-resolves
    # OpenSSL_jll against the upgraded OpenSSL. Only remove the path the
    # installer itself owns, never an externally set PYTHON_JULIAPKG_PROJECT.
    if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/julia_env" ]; then
        rm -rf "${CONDA_PREFIX}/julia_env"
    fi
}

# When Python links OpenSSL < 3.5, juliacall can only resolve its environment on
# Julia <= 1.11, which still ships OpenSSL_jll for the 3.0 series. juliacall
# 0.9.35 supports Julia 1.10.3-1.11 and AGNI runs on 1.11, so when juliaup
# manages Julia this is the non-destructive fix: point the bridge at 1.11.
fix_julia_111_for_openssl() {
    command_exists juliaup || return 1
    info "Switching the juliacall Julia to 1.11 (matches OpenSSL < 3.5)..."
    juliaup add 1.11 </dev/null 2>&1 || return 1
    juliaup default 1.11 </dev/null 2>&1 || return 1
    # Point juliacall at the real 1.11 binary (not the launcher shim) so it can
    # find sys.so, and persist the change for later sessions.
    if command_exists julia; then
        export PYTHON_JULIAPKG_EXE="$(resolve_real_julia_exe)"
        append_export_to_rc "PYTHON_JULIAPKG_EXE" "$PYTHON_JULIAPKG_EXE" "$RC_FILE"
    fi
    # Re-resolve from scratch against Julia 1.11.
    if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/julia_env" ]; then
        rm -rf "${CONDA_PREFIX}/julia_env"
    fi
}

# Verify that "import proteus" works. The import is the first place the Julia
# bridge is loaded, so it surfaces environment problems the pip installs do not.
# Capture the error (the old check discarded it), and self-heal the known conda
# MPI/HDF5 conflict before giving up.
verify_proteus_import() {
    info "Verifying that PROTEUS imports..."
    local import_log
    if import_log=$(python3 -c "import proteus" 2>&1); then
        info "PROTEUS import OK"
        return 0
    fi
    fail "PROTEUS failed to import. Error output:"
    printf '%s\n' "$import_log"
    # An architecture mismatch between Python and Julia cannot be self-healed.
    if printf '%s' "$import_log" | grep -qi 'incompatible architecture'; then
        warn "Python and Julia are built for different CPU architectures (see above)."
        warn "On Apple Silicon, install a native arm64 conda (e.g. miniforge), recreate"
        warn "the proteus env, and re-run install.sh."
        die "PROTEUS Python package failed to import. The full error is above and in $LOGFILE."
    fi
    if printf '%s' "$import_log" | grep -qiE 'libmpi|PMPI_|ompi_|mpich|Symbol not found' \
       || conda_has_mpi_netcdf; then
        warn "This matches the conda MPI/HDF5 conflict. Switching to no-MPI builds and retrying..."
        if fix_conda_mpi_builds; then
            if import_log=$(python3 -c "import proteus" 2>&1); then
                info "PROTEUS import OK after switching to no-MPI builds"
                return 0
            fi
            fail "Still failing after the no-MPI fix. Error output:"
            printf '%s\n' "$import_log"
        fi
    fi
    # Self-heal the Julia / OpenSSL mismatch in the juliacall environment. Julia
    # 1.12 ships OpenSSL_jll 3.5+, so a Python linking OpenSSL < 3.5 cannot
    # resolve it. Two routes: move the bridge to Julia 1.11 (which ships
    # OpenSSL_jll for the 3.0 series), or raise the Python OpenSSL to >= 3.5.
    if printf '%s' "$import_log" | grep -qi 'OpenSSL_jll'; then
        # Prefer the non-destructive route when juliaup manages Julia.
        if python_openssl_below_35 && command_exists juliaup; then
            if fix_julia_111_for_openssl; then
                warn "Moved the juliacall environment to Julia 1.11. Retrying import..."
                if import_log=$(python3 -c "import proteus" 2>&1); then
                    info "PROTEUS import OK on Julia 1.11"
                    return 0
                fi
                fail "Still failing after switching to Julia 1.11. Error output:"
                printf '%s\n' "$import_log"
            fi
        fi
        # Otherwise raise the Python OpenSSL so Julia 1.12 can resolve.
        if conda_openssl_too_old_for_julia && fix_conda_openssl; then
            warn "Upgraded OpenSSL to match the Julia environment. Retrying import..."
            if import_log=$(python3 -c "import proteus" 2>&1); then
                info "PROTEUS import OK after upgrading OpenSSL"
                return 0
            fi
            fail "Still failing after the OpenSSL upgrade. Error output:"
            printf '%s\n' "$import_log"
        fi
        warn "The Julia environment cannot resolve OpenSSL_jll: Julia 1.12 needs OpenSSL >= 3.5"
        warn "in this Python environment. Fix it with either:"
        warn "  juliaup add 1.11 && juliaup default 1.11      (use Julia 1.11 instead)"
        warn "  conda install -c conda-forge 'openssl>=3.5'   (keep Julia 1.12, then re-run)"
    fi
    die "PROTEUS Python package failed to import. The full error is above and in $LOGFILE."
}

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

# juliacall loads Julia into the Python process via dlopen, so Python and Julia
# must share a CPU architecture. The common Apple Silicon mix-up is an x86_64
# (Intel/Rosetta) conda Python with a native arm64 Julia, which resolves the
# Julia environment in a subprocess but then fails to embed with "incompatible
# architecture". Catch it here rather than after the full submodule install.
norm_arch() { case "$1" in arm64|aarch64) echo arm ;; x86_64|amd64) echo x86 ;; *) echo "$1" ;; esac; }
py_arch=$(python3 -c "import platform; print(platform.machine())" 2>/dev/null || true)
jl_arch=$(julia --startup-file=no -e 'print(String(Sys.ARCH))' 2>/dev/null || true)
if [ -n "$py_arch" ] && [ -n "$jl_arch" ] && [ "$(norm_arch "$py_arch")" != "$(norm_arch "$jl_arch")" ]; then
    fail "Python ($py_arch) and Julia ($jl_arch) are built for different CPU architectures."
    warn "juliacall loads Julia into the Python process, so the two must match."
    warn "On Apple Silicon this usually means an Intel (x86_64) conda. Install a native"
    warn "arm64 conda (e.g. miniforge for Apple Silicon), recreate the proteus env, and"
    warn "re-run install.sh. A native arm64 conda also provides OpenSSL >= 3.5."
    die "Python and Julia CPU architectures do not match."
fi

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

# PYTHON_JULIAPKG_EXE. Resolve the real binary so juliacall can find sys.so even
# when julia on PATH is the juliaup launcher shim.
julia_exe="$(resolve_real_julia_exe)"
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

# Detect SSH access to GitHub. A successful key auth returns exit 1 with
# "successfully authenticated"; a missing or rejected key fails so we fall back
# to HTTPS (the submodules are public). Capture the output and exit code up
# front: a later bare $? would report the exit code of an intervening test, not
# of ssh.
# ssh -T returns non-zero even on success (exit 1, "does not provide shell
# access"), so the capture must stay inside an if to survive set -e; a bare
# assignment would abort the script.
if ssh_test_out=$(ssh -T -o BatchMode=yes git@github.com 2>&1); then
    ssh_test_rc=0
else
    ssh_test_rc=$?
fi
if printf '%s' "$ssh_test_out" | grep -qiE 'bad (owner|permissions)|permissions .* are too open'; then
    warn "OpenSSH is ignoring your ~/.ssh files because of their owner or permissions:"
    printf '%s\n' "$ssh_test_out"
    warn "Falling back to HTTPS for the public submodule clones. To use SSH instead,"
    warn "fix the permissions and re-run install.sh:"
    warn "  chmod 700 ~/.ssh"
    warn "  chmod 600 ~/.ssh/config ~/.ssh/id_* 2>/dev/null"
    warn "  chmod 644 ~/.ssh/*.pub ~/.ssh/known_hosts 2>/dev/null"
    warn "If the message says 'Bad owner', also confirm you own them: ls -le ~/.ssh"
    GH_PREFIX="https://github.com/"
elif [ "$ssh_test_rc" -eq 1 ] && printf '%s' "$ssh_test_out" | grep -qi 'successfully authenticated'; then
    GH_PREFIX="git@github.com:"
    info "GitHub SSH access: OK"
else
    GH_PREFIX="https://github.com/"
    info "GitHub SSH access: not available, using HTTPS"
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

# Verify the installation (captures the error and self-heals the conda MPI clash)
verify_proteus_import
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
