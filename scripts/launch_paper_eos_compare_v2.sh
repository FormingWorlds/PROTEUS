#!/usr/bin/env bash
# Launch the 4 v2 WB17-vs-PALEOS paper-comparison runs in parallel,
# detached from the shell so they survive SSH disconnect and the
# 05:00 CET bot restart.
#
# v2 differences from v1 (launch_paper_eos_compare.sh):
#   - 4 runs instead of 8 (single IC mode `liquidus_super` covers
#     both v1 ICs at the EoS level; we vary delta_T_super={100, 500} K).
#   - Both EoS variants use the pre-generated PALEOS-Fei2021 melting
#     curves under FWL_DATA (no Monteux-600 vs PALEOS-derived
#     mismatch).
#
# Each job writes its launch log to output/<name>/launch.log, and its
# proteus_00.log under output/<name>/ as usual. PIDs are written to
# /tmp/paper_eos_compare_v2_pids.txt for monitoring.
#
# Prerequisite (run once per machine that hosts FWL_DATA):
#   python scripts/gen_paleos_melting_curves.py
#
# Usage: bash scripts/launch_paper_eos_compare_v2.sh
set -eo pipefail   # NOT -u: conda's gfortran activate hook references
                   # an unbound variable (GFORTRAN) and would fail.

cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$(pwd)"

# Activate the proteus conda env (Mac Studio default).
eval "$(/Users/timlichtenberg/miniforge3/bin/conda shell.zsh hook)"
conda activate proteus

# Make sure julia is on PATH (Aragog needs it).
export PATH="${CONDA_PREFIX}/bin:${PATH}"

# Use the same FWL_DATA path used by the recent T2.1i, PALEOS validation,
# and v1 paper-comparison runs. Override .zshrc if it points elsewhere.
export FWL_DATA="/Users/timlichtenberg/work/fwl_data"
[ -d "${FWL_DATA}" ] || { echo "FWL_DATA not found: ${FWL_DATA}"; exit 1; }
[ -d "${FWL_DATA}/interior_lookup_tables/1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018_1TPa" ] \
    || { echo "WB17 1TPa tables missing in FWL_DATA"; exit 1; }
[ -f "${FWL_DATA}/interior_lookup_tables/Melting_curves/PALEOS-Fei2021/liquidus_P-T.dat" ] \
    || { echo "PALEOS-Fei2021 melting curves not found; run scripts/gen_paleos_melting_curves.py first"; exit 1; }

# RAD_DIR + FC_DIR (atmosphere modules).
[ -n "${RAD_DIR:-}" ] && [ -d "${RAD_DIR}" ] || export RAD_DIR="${REPO}/socrates"
[ -n "${FC_DIR:-}" ]  && [ -d "${FC_DIR}"  ] || export FC_DIR="${REPO}/AGNI/fastchem"

# Verify python is the env one.
which python | grep -q "envs/proteus/bin/python" \
    || { echo "wrong python: $(which python)"; exit 1; }

CONFIGS=(
    paper_eos_compare_v2_paleos_1me_hot_lqdsupr
    paper_eos_compare_v2_paleos_1me_warm_lqdsupr
    paper_eos_compare_v2_wb17_1me_hot_lqdsupr
    paper_eos_compare_v2_wb17_1me_warm_lqdsupr
)

echo "machine: $(hostname -s) | env: ${CONDA_DEFAULT_ENV} | python: $(which python)"
echo "FWL_DATA=${FWL_DATA}"
echo "RAD_DIR=${RAD_DIR}"
echo "FC_DIR=${FC_DIR}"
echo "launching ${#CONFIGS[@]} runs in parallel..."
echo

PID_FILE="/tmp/paper_eos_compare_v2_pids.txt"
: > "${PID_FILE}"

for cfg in "${CONFIGS[@]}"; do
    outdir="${REPO}/output/${cfg}"
    mkdir -p "${outdir}"
    nohup proteus start --offline -c "input/chili/${cfg}.toml" \
         > "${outdir}/launch.log" 2>&1 &
    pid=$!
    disown "${pid}"
    echo "${cfg} ${pid}" >> "${PID_FILE}"
    printf "  launched %-58s pid=%d\n" "${cfg}" "${pid}"
done

echo
echo "all ${#CONFIGS[@]} launched."
echo "pids in ${PID_FILE}"
echo "monitor with:"
echo "    tail -F output/<name>/proteus_00.log"
echo "    ps -p \$(awk '{print \$2}' ${PID_FILE} | paste -sd, -) -o pid,etime,pcpu,rss,comm"
