#!/usr/bin/env bash
# Install the SUNDIALS CVODE solver for Aragog's production integration path.
#
# Aragog's production interior solver (interior_energetics.aragog.solver_method
# = "cvode") imports CVODE from scikits_odes_sundials, the Python wrapper around
# the SUNDIALS C library. Without it Aragog silently falls back to scipy Radau,
# which is slower and step-size-fragile on multi-Myr coupled cooling runs (it
# trips the core-temperature-jump guard at loose tolerance and stalls at tight
# tolerance). CVODE is the same SUNDIALS integrator SPIDER uses.
#
# This installs the SUNDIALS C library from conda-forge and builds the
# scikits-odes-sundials wrapper against it. It is idempotent: it exits early
# when CVODE already imports. Requires an active conda environment.
set -euo pipefail

if python -c "import scikits_odes_sundials.cvode" >/dev/null 2>&1; then
    echo "CVODE (scikits-odes-sundials) already installed; nothing to do."
    exit 0
fi

if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "ERROR: installing CVODE needs an active conda environment" >&2
    echo "       (CONDA_PREFIX is unset). Activate the proteus env and re-run." >&2
    exit 1
fi

conda_bin="${CONDA_EXE:-conda}"

echo "Installing the SUNDIALS C library (conda-forge) into ${CONDA_PREFIX}..."
"$conda_bin" install -y --prefix "$CONDA_PREFIX" -c conda-forge sundials

echo "Building scikits-odes-sundials against SUNDIALS..."
# The build (scikit-build-core / CMake) locates SUNDIALS through the conda
# prefix; expose it both ways so older and newer build backends find it.
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}:${CMAKE_PREFIX_PATH:-}"
export SUNDIALS_INST="${CONDA_PREFIX}"
pip install scikits-odes-sundials

python -c "import scikits_odes_sundials.cvode" \
    && echo "[+] CVODE (scikits-odes-sundials) installed and importable."
