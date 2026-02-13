#!/usr/bin/env bash
set -e

# This script is only for macOS
if [[ "$(uname)" != "Darwin" ]]; then
  echo "Error: This script is only for macOS. On Linux, use ./tools/get_petsc.sh directly."
  exit 1
fi

# Resolve the PROTEUS root directory relative to this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROTEUS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PETSC_DIR="${PROTEUS_DIR}/petsc"

if [[ ! -d "$PETSC_DIR" ]]; then
  echo "Error: petsc/ directory not found at ${PETSC_DIR}"
  echo "Run ./tools/get_petsc.sh first."
  exit 1
fi

# Check that required tools are available
command -v brew >/dev/null 2>&1 || {
  echo "Error: Homebrew not found. Install it from https://brew.sh/"
  exit 1
}

command -v xcrun >/dev/null 2>&1 || {
  echo "Error: xcrun not found. Install Xcode Command Line Tools: xcode-select --install"
  exit 1
}

for tool in mpicc mpicxx mpifort; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "Error: $tool not found. Install MPI via Homebrew: brew install open-mpi"
    exit 1
  }
done

# Detect Homebrew prefix (Apple Silicon: /opt/homebrew, Intel: /usr/local)
BREW_PREFIX=$(brew --prefix)

# Set compiler environment for this script only (not persisted globally)
export SDKROOT=$(xcrun --show-sdk-path)
export CC=mpicc
export CXX=mpicxx
export FC=mpifort
export F77=mpifort

# Persist SDKROOT in shell config (needed for future builds)
if [[ "$SHELL" == */zsh ]]; then
  MY_RC="$HOME/.zshrc"
elif [[ "$SHELL" == */bash ]]; then
  MY_RC="$HOME/.bashrc"
else
  echo "Error: Could not detect Zsh or Bash."
  echo "Please add the following line to your shell config manually:"
  echo '  export SDKROOT=$(xcrun --show-sdk-path)'
  exit 1
fi

if ! grep -q 'SDKROOT' "$MY_RC" 2>/dev/null; then
  echo "" >> "$MY_RC"
  echo "# PETSc/PROTEUS: macOS SDK path (added by fix_petsc_compile.sh)" >> "$MY_RC"
  echo 'export SDKROOT=$(xcrun --show-sdk-path)' >> "$MY_RC"
  echo "Added SDKROOT to $MY_RC"
else
  echo "SDKROOT already set in $MY_RC, skipping."
fi

# Run the configure command
cd "$PETSC_DIR"
./configure \
   PETSC_ARCH=arch-darwin-c-opt \
   CC=mpicc \
   CXX=mpicxx \
   FC=mpifort \
   F77=mpifort \
   LDFLAGS="-L${BREW_PREFIX}/lib -Wl,-w" \
   --with-debugging=0 \
   --with-cxx-dialect=14 \
   --download-sundials2=1

# Once that finishes successfully, build and test it
make PETSC_DIR=$(pwd) PETSC_ARCH=arch-darwin-c-opt all
make PETSC_DIR=$(pwd) PETSC_ARCH=arch-darwin-c-opt check
