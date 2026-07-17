#!/bin/bash
# Configure petitRADTRANS to read the opacity tables PROTEUS installs.
#
# The package itself comes from the observe extra:
#     pip install -e ".[observe]"
# and the opacity tables come from the FWL data distribution, like every other PROTEUS
# dataset:
#     python -c "from proteus.utils.prt_data import download_prt_opacities; download_prt_opacities()"
#
# This script only writes the configuration file that points petitRADTRANS at those
# tables and pins one default file per species, so the library never stops to ask which
# file to use. The species come from proteus.utils.constants, so adding a gas there
# carries through without editing this script.

set -euo pipefail

echo "Configuring petitRADTRANS..."

python - <<'PY'
import sys

from proteus.utils.prt_data import (
    opacities_present,
    prt_input_data_dir,
    uncovered_species,
    write_prt_config,
)

if not opacities_present():
    sys.exit(
        f"No opacity tables under {prt_input_data_dir()}.\n"
        "Fetch them first:\n"
        "    python -c \"from proteus.utils.prt_data import download_prt_opacities; "
        "download_prt_opacities()\""
    )

# Every species that does have a table is pinned to one file, so petitRADTRANS never
# stops to ask which to use. A species with no table at all is dropped from the transfer
# by the observe module, so it is reported rather than treated as a failure.
path = write_prt_config()
print(f"Wrote {path}")

uncovered = uncovered_species()
if uncovered:
    print(
        "No opacity tables installed for: " + ", ".join(uncovered) + "\n"
        "Spectra will be computed without their contribution."
    )
PY

echo "Done."
