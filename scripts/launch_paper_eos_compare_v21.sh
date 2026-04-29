#!/usr/bin/env bash
# Launch the 4 v2.1 WB17-vs-PALEOS paper-comparison runs in parallel,
# detached from the shell so they survive SSH disconnect and the
# 05:00 CET bot restart.
#
# v2.1 differences from v2:
#   - Same liquidus_super IC, same delta_T_super={100, 500} K, same
#     PALEOS-Fei2021 P-T melting curve config.
#   - WB17 entropy-solver P-S melting tables now DERIVED from the
#     configured PALEOS-Fei2021 P-T file via WB17 EoS T(P,S) inversion,
#     instead of byte-copied from MgSiO3_Wolf_Bower_2018_1TPa/. This
#     closes the v2 bookkeeping leak documented in
#     finding_2026_04_29_v2_melting_curve_mismatch.md.
#   - PALEOS path: same derivation override applies, replacing the
#     Zalmoxis-auto-generated P-S melting tables with derivations from
#     the configured P-T file. Both pipelines now share a single source
#     of truth for the melting curve.
#
# Each job writes its launch log to output/<name>/launch.log, and its
# proteus_00.log under output/<name>/ as usual. PIDs are written to
# /tmp/paper_eos_compare_v21_pids.txt for monitoring.
#
# Usage: bash scripts/launch_paper_eos_compare_v21.sh
set -eo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$(pwd)"

eval "$(/Users/timlichtenberg/miniforge3/bin/conda shell.zsh hook)"
conda activate proteus

export PATH="${CONDA_PREFIX}/bin:${PATH}"
export FWL_DATA="/Users/timlichtenberg/work/fwl_data"
[ -d "${FWL_DATA}" ] || { echo "FWL_DATA not found: ${FWL_DATA}"; exit 1; }
[ -d "${FWL_DATA}/interior_lookup_tables/1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018_1TPa" ] \
    || { echo "WB17 1TPa tables missing in FWL_DATA"; exit 1; }
[ -f "${FWL_DATA}/interior_lookup_tables/Melting_curves/PALEOS-Fei2021/liquidus_P-T.dat" ] \
    || { echo "PALEOS-Fei2021 melting curves not found; run scripts/gen_paleos_melting_curves.py first"; exit 1; }

[ -n "${RAD_DIR:-}" ] && [ -d "${RAD_DIR}" ] || export RAD_DIR="${REPO}/socrates"
[ -n "${FC_DIR:-}" ]  && [ -d "${FC_DIR}"  ] || export FC_DIR="${REPO}/AGNI/fastchem"

which python | grep -q "envs/proteus/bin/python" \
    || { echo "wrong python: $(which python)"; exit 1; }

CONFIGS=(
    paper_eos_compare_v21_paleos_1me_hot_lqdsupr
    paper_eos_compare_v21_paleos_1me_warm_lqdsupr
    paper_eos_compare_v21_wb17_1me_hot_lqdsupr
    paper_eos_compare_v21_wb17_1me_warm_lqdsupr
)

echo "machine: $(hostname -s) | env: ${CONDA_DEFAULT_ENV} | python: $(which python)"
echo "FWL_DATA=${FWL_DATA}"
echo "RAD_DIR=${RAD_DIR}"
echo "FC_DIR=${FC_DIR}"
echo "launching ${#CONFIGS[@]} v2.1 runs in parallel..."
echo

PID_FILE="/tmp/paper_eos_compare_v21_pids.txt"
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
