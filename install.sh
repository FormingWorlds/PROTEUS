#!/usr/bin/env bash
# PROTEUS unified installer; `bash install.sh --help` lists the options. It uses the active conda
# environment, or makes one with Python 3.12 when none is active; it needs the system packages of
# docs/How-to/installation.md, and each phase can run again after a failure.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Python target: fallback values, overridden from pyproject.toml's
# requires-python ceiling once the repo root is known (see below).
REQUIRED_PYTHON_MAJOR=3
REQUIRED_PYTHON_MINOR=12
# Julia: PROTEUS runs on 1.13.x (what CI tests); a default Julia of another
# version stays, and juliaup adds 1.13 beside it.
REQUIRED_JULIA_MAJOR=1
REQUIRED_JULIA_MINOR=13
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
warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*" >&2; }
fail()  { printf "${RED}[FAIL]${NC}  %s\n" "$*" >&2; }
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
    for v in FWL_DATA RAD_DIR FC_DIR PYTHON_JULIAPKG_EXE PYTHON_JULIACALL_BINDIR JULIA_CHANNEL; do
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
        log_file_sentence
        LOG_SENTENCE_SHOWN=true
        echo "If you need help, send that log file to dev@proteus-framework.org,"
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

# The one sentence that names the log file, in die, run_logged and the exit hint.
log_file_sentence() { printf 'The full output of the installer is in %s\n' "$LOGFILE"; }

# Wait, up to 5 s, until the last line of the log holds the text $1: the log tee runs
# behind the script, and run_logged appends a command's output to the log directly.
log_catch_up() {
    local i=0
    [ "${LOG_TEE:-false}" = true ] || return 0
    while [ "$i" -lt 50 ]; do
        case "$(tail -n 1 "$LOGFILE" 2>/dev/null)" in *"$1"*) return 0 ;; esac
        sleep 0.1
        i=$((i + 1))
    done
}

# Copy stdin to the terminal only, past the log tee when it runs ($1: 1 stdout, 2 stderr).
to_terminal() {
    if [ "${LOG_TEE:-false}" != true ]; then
        cat >&"$1"
    elif [ "$1" = 1 ]; then
        cat >&3
    else
        cat >&4
    fi
}

# Usage: run_logged [--takes <duration>] <what is built> <command> [args...]. One line at the
# start and one at the end; the output goes to $LOGFILE only. A failure shows its last 40 lines.
run_logged() {
    local takes="" what rc=0 t0=$SECONDS tmp where="the log file"
    local show='checked out release tag|installed paired aragog branch|stays as it is|^    (From the PROTEUS folder|Or replace it)|^        (git -C|pip install -e|bash tools/)'
    if [ "${1:-}" = "--takes" ]; then
        takes=" (can take $2)"
        shift 2
    fi
    what="$1"
    shift
    # Without a writable log the command runs as it is, its output on the terminal.
    if [ -z "${LOGFILE:-}" ] || ! { : >> "$LOGFILE"; } 2>/dev/null \
       || ! tmp=$(mktemp "${TMPDIR:-/tmp}/proteus-install.XXXXXX" 2>/dev/null); then
        "$@" || rc=$?
        return "$rc"
    fi
    if [ "${RUN_LOGGED_TOLD:-false}" != true ]; then
        where="$LOGFILE"
        RUN_LOGGED_TOLD=true
    fi
    if [ "${VERBOSE:-false}" = true ]; then
        info "$what$takes..."
        if [ "${LOG_TEE:-false}" = true ]; then
            "$@" || rc=$?
        elif "$@" 2>&1 | tee -a "$LOGFILE"; then
            rc=0
        else
            rc=${PIPESTATUS[0]}
        fi
    else
        info "$what$takes; its output goes to $where"
        log_catch_up "$what$takes; its output goes to $where"
        # stdin is /dev/null: a step whose output is hidden must not wait for an answer.
        if "$@" </dev/null 2>&1 | tee -a "$LOGFILE" > "$tmp"; then
            rc=0
        else
            rc=${PIPESTATUS[0]}
        fi
    fi
    if [ "$rc" -eq 0 ]; then
        grep -E "$show" "$tmp" | to_terminal 1 || true
        info "$what: done in $((SECONDS - t0)) s"
    else
        fail "$what: failed with exit status $rc after $((SECONDS - t0)) s"
        if [ "${VERBOSE:-false}" != true ]; then
            log_catch_up "$what: failed with exit status $rc"
            echo "--- the last 40 lines of its output:" | to_terminal 2
            tail -n 40 "$tmp" | to_terminal 2
        fi
        log_file_sentence >&2
        LOG_SENTENCE_SHOWN=true
    fi
    rm -f "$tmp"
    return "$rc"
}

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

# juliacall needs the real Julia binary as PYTHON_JULIAPKG_EXE, not the juliaup launcher
# shim, to find sys.so beside it. A "+1.13" channel selector goes to the launcher beside
# juliaup; without one, a failed probe falls back to the julia on PATH.
resolve_real_julia_exe() {
    local sel="${1:-}" launcher=julia bindir
    if [ -n "$sel" ] && command_exists juliaup \
       && [ -x "$(dirname "$(command -v juliaup)")/julia" ]; then
        launcher="$(dirname "$(command -v juliaup)")/julia"
    fi
    bindir="$("$launcher" $sel --startup-file=no -e 'print(Sys.BINDIR)' </dev/null 2>/dev/null || true)"
    if [ -n "$bindir" ] && [ -x "$bindir/julia" ]; then
        printf '%s\n' "$bindir/julia"
    elif [ -z "$sel" ]; then
        command -v julia 2>/dev/null || true
    fi
}

# The juliaup channel selector of the Julia PROTEUS uses: empty for the default channel.
JULIA_CHANNEL=""

# Point PROTEUS at the real binary of channel $1 (JULIA_CHANNEL, PYTHON_JULIAPKG_EXE);
# returns 1 and changes nothing when the channel has no binary.
use_julia_channel() {
    local exe
    exe="$(resolve_real_julia_exe "$1")"
    [ -n "$exe" ] || return 1
    JULIA_CHANNEL="$1"
    export PYTHON_JULIAPKG_EXE="$exe"
}

# Remove the installer-owned juliacall project so the next import re-resolves
# the Julia environment. Only the path the installer itself owns, never an
# externally set PYTHON_JULIAPKG_PROJECT.
reset_julia_env() {
    if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/julia_env" ]; then
        rm -rf "${CONDA_PREFIX}/julia_env"
    fi
}

# juliacall builds a Julia environment whose OpenSSL_jll is matched to the
# OpenSSL the Python interpreter links against. Julia 1.12 and newer provide
# OpenSSL_jll 3.5 and newer only, so a Python interpreter linking OpenSSL < 3.5
# pins OpenSSL_jll to the 3.0 series and leaves the resolve unsatisfiable.
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
    info "Upgrading OpenSSL to >= 3.5 (required by Julia 1.12 and newer)..."
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
    reset_julia_env
}

# With a Python linking OpenSSL < 3.5, juliacall resolves its environment only on
# Julia <= 1.11 (OpenSSL_jll 3.0 series), so point only PYTHON_JULIAPKG_EXE at a
# juliaup 1.11 binary; the default Julia and the Julia on PATH stay as they are.
fix_julia_111_for_openssl() {
    command_exists juliaup || return 1
    info "Pointing the juliacall Julia at 1.11 (matches OpenSSL < 3.5)..."
    juliaup add 1.11 </dev/null 2>&1 || return 1
    if ! use_julia_channel '+1.11'; then
        warn "Could not locate the Julia 1.11 binary after 'juliaup add 1.11'."
        return 1
    fi
    set_install_var PYTHON_JULIAPKG_EXE "$PYTHON_JULIAPKG_EXE"
    reset_julia_env
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
        die "PROTEUS Python package failed to import. The full error is above."
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
    # 1.12 and newer ship OpenSSL_jll 3.5+, so a Python linking OpenSSL < 3.5 cannot
    # resolve it. Two routes: move the bridge to Julia 1.11 (which ships
    # OpenSSL_jll for the 3.0 series), or raise the Python OpenSSL to >= 3.5.
    if printf '%s' "$import_log" | grep -qi 'OpenSSL_jll'; then
        # Prefer the non-destructive route when juliaup manages Julia.
        if python_openssl_below_35 && command_exists juliaup; then
            if fix_julia_111_for_openssl; then
                warn "Moved the juliacall environment to Julia 1.11. Retrying import..."
                if import_log=$(python3 -c "import proteus" 2>&1); then
                    info "PROTEUS import OK on Julia 1.11"
                    warn "Julia 1.11 is deprecated and support will be dropped in a future release."
                    warn "  OpenSSL >= 3.5 in this environment removes the need for it:"
                    warn "  conda install -c conda-forge 'openssl>=3.5'"
                    return 0
                fi
                fail "Still failing after switching to Julia 1.11. Error output:"
                printf '%s\n' "$import_log"
            fi
        fi
        # Otherwise raise the Python OpenSSL so Julia 1.12 or newer can resolve.
        if conda_openssl_too_old_for_julia && fix_conda_openssl; then
            warn "Upgraded OpenSSL to match the Julia environment. Retrying import..."
            if import_log=$(python3 -c "import proteus" 2>&1); then
                info "PROTEUS import OK after upgrading OpenSSL"
                return 0
            fi
            fail "Still failing after the OpenSSL upgrade. Error output:"
            printf '%s\n' "$import_log"
        fi
        warn "The Julia environment cannot resolve OpenSSL_jll: Julia >= 1.12 needs OpenSSL >= 3.5"
        warn "in this Python environment. Fix it with either:"
        warn "  conda install -c conda-forge 'openssl>=3.5'   (keep your Julia, then re-run)"
        warn "  juliaup add 1.11, then set PYTHON_JULIAPKG_EXE to the path that"
        warn "  julia +1.11 -e 'print(joinpath(Sys.BINDIR, \"julia\"))' prints   (Julia 1.11, deprecated)"
    fi
    die "PROTEUS Python package failed to import. The full error is above."
}

# Export NAME=value ($1, $2) in this process and record it in the exports file, one
# line per variable, written in place; bind_install_vars passes it to the conda env.
set_install_var() {
    local name="$1" value="$2" kept=""
    export "$name=$value"
    if [ -f "$PROTEUS_ENV_FILE" ]; then
        kept=$(grep -v "^export $name=" "$PROTEUS_ENV_FILE" || true)
    fi
    {
        [ -z "$kept" ] || printf '%s\n' "$kept"
        printf "export %s='%s'\n" "$name" "$(printf '%s' "$value" | sed "s/'/'\\\\''/g")"
    } > "$PROTEUS_ENV_FILE"
    case " $INSTALL_VAR_NAMES " in
        *" $name "*) ;;
        *) INSTALL_VAR_NAMES="${INSTALL_VAR_NAMES:+$INSTALL_VAR_NAMES }$name" ;;
    esac
    info "Set $name=$value (recorded in $PROTEUS_ENV_FILE)"
}

# Bind every recorded variable to the active conda env in one call. A pixi env is not
# a conda env, so its variables stay in the exports file only.
bind_install_vars() {
    local name pairs=()
    [ "$CONDA_BIND" = "pending" ] || return 0
    CONDA_BIND="none"
    if [ -z "${CONDA_PREFIX:-}" ] || [ -n "${PIXI_ENVIRONMENT_NAME:-}" ] \
       || [ -z "$INSTALL_VAR_NAMES" ]; then
        return 0
    fi
    for name in $INSTALL_VAR_NAMES; do
        pairs+=("$name=${!name}")
    done
    if command_exists conda \
       && conda env config vars set -p "$CONDA_PREFIX" "${pairs[@]}" </dev/null 2>&1; then
        CONDA_BIND="bound"
        info "Bound $INSTALL_VAR_NAMES to the conda env at $CONDA_PREFIX"
    else
        warn "Could not bind $INSTALL_VAR_NAMES to the conda env at $CONDA_PREFIX."
        warn "  They are in $PROTEUS_ENV_FILE; load them with: source $PROTEUS_ENV_FILE"
    fi
}

# Read only: name each export line of the 4 install variables that an earlier
# install left in a shell start file; the installer does not change these files.
note_old_shell_exports() {
    local f hits hit found=""
    for f in "$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.profile" "$HOME/.zprofile"; do
        [ -f "$f" ] && [ -r "$f" ] || continue
        hits=$(grep -nE '^export (FWL_DATA|RAD_DIR|PYTHON_JULIAPKG_EXE|FC_DIR)=' "$f" || true)
        [ -n "$hits" ] || continue
        while IFS= read -r hit; do
            found+="  $f, line ${hit%%:*}: ${hit#*:}"$'\n'
        done <<< "$hits"
    done
    [ -n "$found" ] || return 0
    warn "An earlier install set these variables in a shell start file:"
    printf '%s' "$found" >&2
    warn "  PROTEUS takes the values from the conda env (or $PROTEUS_ENV_FILE) instead."
    warn "  You can delete those lines by hand; while they stay, a shell without the"
    warn "  activated env still gets the old values."
}

# Build FastChem in the AGNI folder. Without FC_DIR and with no answer on stdin AGNI's script
# asks nothing and writes no shell start file; it then exits 1 after a good build, so the
# built program decides, not that status.
build_fastchem() {
    env -u FC_DIR bash src/get_fastchem.sh </dev/null || true
    [ -x fastchem/fastchem ]
}

# The command that activates the conda env afresh: the user's shell has no env active when
# setup_conda_env activated it in this process only.
conda_activate_command() {
    if [ "${CONDA_ACTIVATED_HERE:-false}" = true ]; then
        echo "conda activate ${CONDA_DEFAULT_ENV:-proteus}"
    else
        echo "conda deactivate && conda activate ${CONDA_DEFAULT_ENV:-proteus}"
    fi
}

# The closing instruction: activate the conda env that holds the variables, or
# source the exports file where no conda env holds them.
print_next_steps() {
    echo "Next steps:"
    if [ "$CONDA_BIND" = "bound" ]; then
        echo "  1. Activate the env, so that it loads the variables:"
        echo "     $(conda_activate_command)"
    elif [ "${CONDA_ACTIVATED_HERE:-false}" = true ]; then
        echo "  1. Activate the env and load the variables, in the PROTEUS folder ($SCRIPT_DIR):"
        echo "     conda activate ${CONDA_DEFAULT_ENV:-proteus} && source .proteus-env.sh"
    else
        echo "  1. Load the variables, in the PROTEUS folder ($SCRIPT_DIR):"
        echo "     source .proteus-env.sh"
    fi
    echo "  2. Run the quick-start tutorial:"
    echo "     proteus start --offline -c input/dummy.toml"
}

# On a failed run, name the log file (unless die or run_logged did) and the exports file.
print_exit_hint() {
    local rc=$?
    if [ "$rc" -ne 0 ] && [ "${LOG_SENTENCE_SHOWN:-false}" != true ] \
       && [ -n "${LOGFILE:-}" ] && [ -f "$LOGFILE" ]; then
        echo ""
        log_file_sentence
    fi
    if [ "$rc" -ne 0 ] && [ -n "$INSTALL_VAR_NAMES" ]; then
        echo ""
        echo "--- NOTE: the variables set so far are in $PROTEUS_ENV_FILE. To keep them on"
        echo "    a re-run, run 'source $PROTEUS_ENV_FILE' first."
    fi
    if [ "$rc" -ne 0 ] && [ "${CONDA_ACTIVATED_HERE:-false}" = true ]; then
        echo ""
        echo "--- NOTE: the installer activated the conda env ${CONDA_DEFAULT_ENV:-proteus} in its own"
        echo "    process only. To use it in your shell, run '$(conda_activate_command)'."
    fi
}

# Get available disk space in GB (POSIX-portable)
available_disk_gb() {
    df -k . | awk 'NR==2 {printf "%d\n", $4/1024/1024}'
}

CURRENT_PHASE=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERACTIVE="false"
VERBOSE="false"

# The log state of run_logged: whether the log tee runs, whether it named the log file.
LOG_TEE="false"
RUN_LOGGED_TOLD="false"
LOG_SENTENCE_SHOWN="false"

# The exports file of set_install_var, and the state that bind_install_vars reads.
PROTEUS_ENV_FILE="$SCRIPT_DIR/.proteus-env.sh"
INSTALL_VAR_NAMES=""
CONDA_BIND="pending"
# The conda env --conda-env names, and whether setup_conda_env activated an env in this process.
CONDA_ENV_REQUESTED=""
CONDA_ACTIVATED_HERE="false"
trap print_exit_hint EXIT

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

DATA_MODE="essential"
while [ "$#" -gt 0 ]; do
    arg="$1"
    shift
    case "$arg" in
        --all-data) DATA_MODE="all" ;;
        --no-data)  DATA_MODE="none" ;;
        --interactive|-i) INTERACTIVE="true" ;;
        --verbose)  VERBOSE="true" ;;
        --conda-env)
            if [ "$#" -eq 0 ] || [ -z "$1" ] || [ "${1#-}" != "$1" ]; then
                fail "--conda-env needs the name of a conda environment."
                echo "Run 'bash install.sh --help' for usage."
                exit 1
            fi
            CONDA_ENV_REQUESTED="$1"
            shift
            ;;
        --help|-h)
            echo "PROTEUS unified installer"
            echo ""
            echo "Usage: bash install.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --all-data   Download all reference data (~10-20 GB) instead of the essential set"
            echo "  --no-data    Skip data downloads entirely, including the essential set"
            echo "  -i, --interactive  Interactive mode (prompt for choices)"
            echo "  --verbose    Show the full build output on the terminal, not only in the log file"
            echo "  --conda-env <name>  The conda environment to use, or to make, when none or base"
            echo "               is active (default: proteus)"
            echo "  --help       Show this message"
            echo ""
            echo "Default: download the essential data (~3 GB): the default-config and tutorial"
            echo "k-tables, stellar spectra and tracks, the PHOENIX grid of the default star"
            echo "([Fe/H] = 0.0, [alpha/M] = 0.0) and the interior tables"
            echo ""
            echo "Each build step prints one line on the terminal; its full output goes to"
            echo "install_<date>_<time>.log in the PROTEUS folder."
            exit 0
            ;;
        *)
            fail "Unknown argument: $arg"
            echo "Run 'bash install.sh --help' for usage."
            exit 1
            ;;
    esac
done

# Start logging; fds 3 and 4 keep the terminal for the lines run_logged shows there only.
LOGFILE="$SCRIPT_DIR/install_$(date +%Y%m%d_%H%M%S).log"
exec 3>&1 4>&2
exec > >(tee -a "$LOGFILE") 2>&1
LOG_TEE="true"

echo ""
printf "${BOLD}PROTEUS Installer${NC}\n"
echo "Log: $LOGFILE"
echo "Data mode: $DATA_MODE"
if [ "$INTERACTIVE" = "true" ]; then
    echo "Mode: interactive (-i)"
fi
if [ "$VERBOSE" = "true" ]; then
    echo "Build output: on the terminal and in the log (--verbose)"
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
# (e.g. ">=3.12,<3.13" targets 3.12), so the installer and the package
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

# Stop with the 2 commands that make and activate the conda env $2 by hand; $1 names the step
# of setup_conda_env that failed.
stop_without_conda_env() {
    fail "No conda environment other than base is active."
    fail "$1"
    echo ""
    echo "Create and activate a conda environment first:"
    echo "  conda create -n $2 python=$REQUIRED_PYTHON_MAJOR.$REQUIRED_PYTHON_MINOR"
    echo "  conda activate $2"
    die "Conda environment required."
}

# Use the active conda env; when none or base is active, use or make the env that --conda-env
# names (default proteus) and activate it in this process. A pixi env is used as it is.
setup_conda_env() {
    local name="${CONDA_ENV_REQUESTED:-proteus}" active="${CONDA_DEFAULT_ENV:-}"
    local pyver="$REQUIRED_PYTHON_MAJOR.$REQUIRED_PYTHON_MINOR" envs base made step="" py
    if [ -n "${PIXI_ENVIRONMENT_NAME:-}" ]; then
        info "Pixi environment: $PIXI_ENVIRONMENT_NAME"
        return 0
    fi
    if [ -n "$active" ] && [ "$active" != base ]; then
        if [ -n "$CONDA_ENV_REQUESTED" ] && [ "$CONDA_ENV_REQUESTED" != "$active" ]; then
            die "--conda-env $CONDA_ENV_REQUESTED and the active conda environment $active disagree; deactivate $active or leave out --conda-env."
        fi
        info "Conda environment: $active"
        return 0
    fi
    if [ "$name" = base ]; then
        die "The installer does not install into the conda base environment; name another with --conda-env."
    fi
    command_exists conda || stop_without_conda_env "conda is not on PATH." "$name"
    if ! envs=$(conda env list </dev/null); then
        stop_without_conda_env "'conda env list' failed." "$name"
    fi
    # An env name is the first column of its line; an env without a name shows only its path.
    if printf '%s\n' "$envs" | awk -v n="$name" '$1 == n {f = 1} END {exit !f}'; then
        made="found"
    else
        info "Creating the conda environment $name with Python $pyver..."
        if ! conda create -y -n "$name" "python=$pyver" </dev/null; then
            stop_without_conda_env "'conda create -y -n $name python=$pyver' failed." "$name"
        fi
        made="made"
    fi
    base=$(conda info --base </dev/null) || base=""
    # conda's activation scripts read unset variables, so set -u is off for these 2 calls.
    set +u
    if [ -z "$base" ] || ! source "$base/etc/profile.d/conda.sh"; then
        step="Loading conda's shell functions from '${base:-?}/etc/profile.d/conda.sh' failed."
    elif ! conda activate "$name"; then
        step="'conda activate $name' failed."
    fi
    set -u
    [ -z "$step" ] || stop_without_conda_env "$step" "$name"
    hash -r
    py=$(command -v python || true)
    case "${CONDA_DEFAULT_ENV:-}|$py" in
        "$name|${CONDA_PREFIX:-?}"/*) ;;
        *) stop_without_conda_env "'conda activate $name' left the env ${CONDA_DEFAULT_ENV:-(none)} active and python at ${py:-(none)}." "$name" ;;
    esac
    # Later conda calls go to the executable, as with an env of the user's: the shell
    # function reactivates the env after an install, and that reads unset variables.
    unset -f conda
    CONDA_ACTIVATED_HERE="true"
    info "Conda environment: $name ($made and activated by the installer for this run)"
}

setup_conda_env

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

# NetCDF (needed for SOCRATES, which uses the C and the Fortran interface)
if ! command_exists nc-config; then
    missing_deps+=("netcdf-dev")
fi
if ! command_exists nf-config; then
    missing_deps+=("netcdf-fortran-dev")
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
            echo "  sudo dnf install gcc-gfortran gcc-c++ netcdf-fortran-devel make curl git cmake unzip"
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

# Print the version that the Julia binary $1 reports, or nothing when it reports none.
julia_version_of() {
    "$1" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1 || true
}

# The one line that names the Julia PROTEUS uses; $1 is the version of a default Julia that
# differs from it, "none" for a juliaup without julia on PATH, or empty.
report_proteus_julia() {
    local version used
    version=$(julia_version_of "$PYTHON_JULIAPKG_EXE")
    used="Julia ${version:-(version unknown)} at $PYTHON_JULIAPKG_EXE"
    case "$1" in
        "") info "PROTEUS uses $used." ;;
        none) info "PROTEUS uses $used; the default Julia of juliaup stays as it is." ;;
        *) info "PROTEUS uses $used; your default Julia $1 stays as it is." ;;
    esac
}

# Add the required Julia beside the default ($1, as for report_proteus_julia) without
# changing the default, and put its binary first on the installer's own PATH.
add_julia_beside_default() {
    local req="$REQUIRED_JULIA_MAJOR.$REQUIRED_JULIA_MINOR"
    info "Adding Julia $req via juliaup, beside the default Julia..."
    if ! juliaup add "$req" </dev/null; then
        warn "'juliaup add $req' failed; looking for an installed $req channel."
    fi
    if ! use_julia_channel "+$req"; then
        die "Could not locate the Julia $req binary after 'juliaup add $req'."
    fi
    PATH="$(dirname "$PYTHON_JULIAPKG_EXE"):$PATH"
    export PATH
    report_proteus_julia "$1"
}

# Install juliaup with the official installer. It has no earlier default to
# protect, so the required Julia becomes its default.
install_fresh_juliaup() {
    local req="$REQUIRED_JULIA_MAJOR.$REQUIRED_JULIA_MINOR" julia_installer
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
    juliaup add "$req"
    juliaup default "$req"
    info "Julia $req installed"
}

# Select the Julia PROTEUS uses, always the required one, and export it as PYTHON_JULIAPKG_EXE;
# only a juliaup that install_fresh_juliaup installed in this run gets a new default.
setup_julia() {
    local req="$REQUIRED_JULIA_MAJOR.$REQUIRED_JULIA_MINOR" julia_version
    if ! command_exists julia && [ -x "$HOME/.juliaup/bin/juliaup" ]; then
        export PATH="$HOME/.juliaup/bin:$PATH"
    fi
    if ! command_exists julia; then
        info "Julia not found."
        if command_exists juliaup; then
            add_julia_beside_default none
        elif prompt_yn "Install Julia via juliaup (official installer)?"; then
            install_fresh_juliaup
            use_julia_channel ""
            report_proteus_julia ""
        else
            die "Julia is required for AGNI. Install it manually: https://julialang.org/downloads/"
        fi
        return 0
    fi
    julia_version=$(julia_version_of julia)
    if [ -z "$julia_version" ]; then
        die "Could not parse Julia version from 'julia --version' output."
    fi
    case "$julia_version" in
        "$req" | "$req".*)
            use_julia_channel ""
            report_proteus_julia ""
            return 0
            ;;
    esac
    if ! command_exists juliaup; then
        warn "PROTEUS runs on Julia $req, found $julia_version."
        warn "juliaup not found. Cannot add Julia $req automatically."
        echo "  Install juliaup: curl -fsSL https://install.julialang.org | sh"
        echo "  Then: juliaup add $req"
        die "Julia version mismatch."
    fi
    add_julia_beside_default "$julia_version"
}

setup_julia

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
# Only flag a recognized arm-vs-x86 mismatch; an unrecognized arch fails open.
case "$(norm_arch "$py_arch") $(norm_arch "$jl_arch")" in
    "arm x86"|"x86 arm")
        fail "Python ($py_arch) and Julia ($jl_arch) are built for different CPU architectures."
        warn "juliacall loads Julia into the Python process, so the two must match. Either:"
        warn "  - use a Python matching Julia ($jl_arch): on Apple Silicon, a native arm64"
        warn "    conda-forge env (miniforge), which also provides OpenSSL >= 3.5; or"
        warn "  - install a Julia matching Python ($py_arch) via juliaup."
        die "Python and Julia CPU architectures do not match."
        ;;
esac

# ===================================================================
# Phase 3: Environment variables
# ===================================================================
CURRENT_PHASE=3
phase 3 "Environment variables"

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
set_install_var FWL_DATA "$fwl_path"

# PYTHON_JULIAPKG_EXE, the real Julia binary that setup_julia selected in Phase 2
set_install_var PYTHON_JULIAPKG_EXE "$PYTHON_JULIAPKG_EXE"

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
    set_install_var RAD_DIR "$RAD_DIR"
else
    # On HPC clusters conda's netcdf can shadow the system one, so the Fortran linker
    # does not see the conda lib/ during the build; Python keeps the conda env.
    _saved_conda_prefix="${CONDA_PREFIX:-}"
    if [ -n "$_saved_conda_prefix" ] && [ -d "$_saved_conda_prefix/lib" ]; then
        _saved_ld="${LD_LIBRARY_PATH:-}"
        export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v "$_saved_conda_prefix" | tr '\n' ':' | sed 's/:$//')
        _saved_library="${LIBRARY_PATH:-}"
        export LIBRARY_PATH=$(echo "${LIBRARY_PATH:-}" | tr ':' '\n' | grep -v "$_saved_conda_prefix" | tr '\n' ':' | sed 's/:$//')
        info "Temporarily hiding conda lib paths from Fortran linker"
    fi
    run_logged --takes "several minutes" "Building SOCRATES" bash tools/get_socrates.sh
    # Restore conda paths
    if [ -n "$_saved_conda_prefix" ] && [ -d "$_saved_conda_prefix/lib" ]; then
        export LD_LIBRARY_PATH="$_saved_ld"
        export LIBRARY_PATH="$_saved_library"
    fi
    # Verify compilation succeeded (radlib.a is the primary build artifact)
    if [ -d "$SCRIPT_DIR/socrates" ] && [ -f "$SCRIPT_DIR/socrates/bin/radlib.a" ]; then
        set_install_var RAD_DIR "$SCRIPT_DIR/socrates"
        info "SOCRATES installed, RAD_DIR=$RAD_DIR"
    else
        die "SOCRATES compilation failed: socrates/bin/radlib.a was not built."
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
    run_logged --takes "several minutes" "Installing AGNI and its Julia packages" \
        bash tools/get_agni.sh 0
    if [ ! -f "$SCRIPT_DIR/AGNI/Manifest.toml" ]; then
        die "AGNI installation failed (Julia packages not resolved)."
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
    cd "$AGNI_REAL"
    # If a previous NFS-failed clone left a partial fastchem dir, clean it
    if [ -d "$AGNI_REAL/fastchem/.git" ] && [ ! -f "$AGNI_REAL/fastchem/CMakeLists.txt" ]; then
        warn "Cleaning up partial FastChem clone from a previous failed attempt"
        rm -rf "$AGNI_REAL/fastchem"
    fi
    run_logged "Building FastChem" build_fastchem || true
    cd "$SCRIPT_DIR"
    if [ ! -x "$AGNI_REAL/fastchem/fastchem" ]; then
        warn "FastChem installation failed. AGNI chemistry features will not work."
        warn "Install manually: cd AGNI && bash src/get_fastchem.sh"
    else
        info "FastChem installed"
    fi
fi

# FC_DIR, on the resolved AGNI path so that it survives a change of the symlink
if [ -d "$AGNI_REAL/fastchem" ]; then
    set_install_var FC_DIR "$AGNI_REAL/fastchem"
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

# Usage: clone_and_install <RepoName> <distribution>. tools/get_python_module.sh clones a missing module
# at its release tag, or installs an existing folder as it is and names its state; it never replaces one.
clone_and_install() {
    local repo="$1" dist="$2"
    if ! run_logged "Setting up $repo" env PROTEUS_MODULE_GIT_BASE="$GH_PREFIX" \
        bash "$SCRIPT_DIR/tools/get_python_module.sh" "$repo"; then
        die "Could not install $repo ($dist); see the error above"
    fi
}

# Print one line naming the module clones that are not at their release tag, which record_clone_stays
# (tools/_get_common.sh) wrote to the file PROTEUS_CLONE_STAYS names, and remove that file.
report_stayed_clones() {
    local list
    [ -n "${PROTEUS_CLONE_STAYS:-}" ] || return 0
    if [ -s "$PROTEUS_CLONE_STAYS" ]; then
        list=$(awk 'NR > 1 {printf ", "} {printf "%s", $0}' "$PROTEUS_CLONE_STAYS")
        warn "Not at their release tag: $list"
    fi
    rm -f "$PROTEUS_CLONE_STAYS"
}

PROTEUS_CLONE_STAYS=$(mktemp "${TMPDIR:-/tmp}/proteus-clone-stays.XXXXXX") || PROTEUS_CLONE_STAYS=""
export PROTEUS_CLONE_STAYS
clone_and_install "MORS"     "fwl-mors"
clone_and_install "JANUS"    "fwl-janus"
clone_and_install "CALLIOPE" "fwl-calliope"
clone_and_install "ZEPHYRUS" "fwl-zephyrus"

# Aragog and Zalmoxis use dedicated setup scripts
run_logged "Setting up Aragog" bash tools/get_aragog.sh
run_logged "Setting up Zalmoxis" bash tools/get_zalmoxis.sh
report_stayed_clones

# Install the SUNDIALS CVODE solver, Aragog's default integrator. A run with
# Aragog stops at setup without it.
run_logged "Setting up the SUNDIALS CVODE solver" bash tools/get_cvode.sh \
    || warn "CVODE install failed. Aragog with solver_method = \"cvode\" stops at setup until it imports; fix the error above and re-run bash tools/get_cvode.sh. Other interior modules, radau and bdf do not need it."

# Install PROTEUS itself
run_logged --takes "a few minutes" "Installing PROTEUS and its remaining dependencies" \
    pip install -e ".[develop]"

# The PROTEUS install can replace packages; Aragog on "cvode" needs CVODE to still import.
python -c "from scikits_odes_sundials.cvode import CVODE, CV_RootFunction, StatusEnum" >/dev/null 2>&1 \
    || warn "CVODE does not import after the PROTEUS install. Aragog with solver_method = \"cvode\" stops at setup until it does; re-run bash tools/get_cvode.sh."

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
        info "Downloading essential reference data (~3 GB)..."
        # Default-config and tutorial k-tables only; the bare
        # `proteus get spectral` downloads every group (~10 GB).
        # Honeyside/48 serves the config default (src/proteus/config/_atmos_clim.py),
        # Dayspring/48 serves input/tutorials/*.toml; update here when those change.
        proteus get spectral -n Honeyside -b 48 2>&1 || warn "Spectral data download had issues"
        proteus get spectral -n Dayspring -b 48 2>&1 || warn "Spectral data download had issues"
        proteus get stellar 2>&1 || warn "Stellar data download had issues"
        # The default star (src/proteus/config/_star.py) reads this PHOENIX grid (~1 GB on disk).
        proteus get phoenix --feh +0.0 --alpha +0.0 2>&1 || warn "PHOENIX download had issues"
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
bind_install_vars
echo ""
printf "${GREEN}${BOLD}Installation complete!${NC}\n"
echo ""
note_old_shell_exports
print_next_steps
echo ""
echo "For SPIDER (legacy interior module), run separately:"
echo "  bash tools/get_petsc.sh && bash tools/get_spider.sh"
echo ""
echo "Full documentation: https://proteus-framework.org/PROTEUS/"
echo "Log saved to: $LOGFILE"
