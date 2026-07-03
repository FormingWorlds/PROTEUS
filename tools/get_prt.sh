#!/bin/bash
# Download and install petitRADTRANS

# Load shell-defined environment variables such as FWL_DATA.
if [ -z "${FWL_DATA:-}" ] && [ -f "$HOME/.bashrc" ]; then
    # shellcheck disable=SC1090
    source "$HOME/.bashrc"
fi

echo "Installing petitRADTRANS into Python environment..."

# Installing meson first
python -m pip install numpy meson meson-python ninja

# Make room
workpath="prt/"
rm -rf $workpath

# Download
echo "Cloning from GitHub"
uri="https://github.com/FormingWorlds/petitRADTRANS.git"
echo "    $uri -> $workpath"
git clone "$uri" "$workpath"

# Change dir and install
olddir=$(pwd)
cd $workpath
python -m pip install -U -e . --no-build-isolation

python - <<'PY'
from pathlib import Path
import shutil

targets = [
    Path('petitRADTRANS/__file_conversion.py'),
    Path('petitRADTRANS/opacities/opacities.py'),
]

needle = '    except ImportError:\n'
replacement = '    except Exception:\n'

for path in targets:
    text = path.read_text(encoding='utf-8')
    shutil.copy2(path, Path(str(path) + '.bak'))
    idx = text.find(needle)
    if idx != -1:
        text = text[:idx] + replacement + text[idx + len(needle) :]
    path.write_text(text, encoding='utf-8')
PY

echo "Installing petitRADTRANS opacity tables..."
prt_input_data_path="$(pwd)/input_data"
if [ -n "${FWL_DATA:-}" ]; then
    prt_input_data_path="${FWL_DATA%/}/prt/input_data"
fi
config_file="$HOME/.petitradtrans/petitradtrans_config_file.ini"
if [ -f "$config_file" ]; then
    mv "$config_file" "${config_file}.bak"
fi
rm -rf ~/.petitradtrans/petitradtrans_config_file.ini

python - <<'PY'
from petitRADTRANS.radtrans import Radtrans

Radtrans(line_species=['Ti_+'])
PY

export PRT_INPUT_DATA_PATH="$prt_input_data_path"
python - <<'PY'
from pathlib import Path
import os
import shutil

config_path = Path.home() / '.petitradtrans' / 'petitradtrans_config_file.ini'
text = config_path.read_text(encoding='utf-8')
shutil.copy2(config_path, Path(str(config_path) + '.bak'))

# Update prt_input_data_path line.
new_lines = []
for line in text.splitlines(keepends=True):
    if line.startswith('prt_input_data_path = '):
        new_lines.append(f"prt_input_data_path = {os.environ['PRT_INPUT_DATA_PATH']}\n")
    else:
        new_lines.append(line)
text = ''.join(new_lines)

# Insert opacity aliases before [Paths] section, matching previous sed behavior.
aliases = [
    'opacities/lines/correlated_k/H2O/1H2-16O = 1H2-16O__HITEMP.R1000_0.1-250mu.ktable.petitRADTRANS.h5\n',
    'opacities/lines/correlated_k/TiO/48Ti-16O = 48Ti-16O__McKemmish.R1000_0.1-250mu.ktable.petitRADTRANS.h5\n',
    'opacities/lines/correlated_k/VO/51V-16O = 51V-16O__Plez.R1000_0.1-250mu.ktable.petitRADTRANS.h5\n',
    'opacities/lines/correlated_k/Na/23Na = 23Na__Allard.R1000_0.1-250mu.ktable.petitRADTRANS.h5\n',
    'opacities/lines/correlated_k/K/39K = 39K__Allard.R1000_0.1-250mu.ktable.petitRADTRANS.h5\n',
    'opacities/lines/correlated_k/CH4/12C-1H4 = 12C-1H4__HITEMP.R1000_0.1-250mu.ktable.petitRADTRANS.h5\n',
    'opacities/continuum/collision_induced_absorptions/H2--H2/H2--H2-NatAbund = H2--H2-NatAbund__BoRi.R831_0.6-250mu.ciatable.petitRADTRANS.h5\n',
    'opacities/continuum/collision_induced_absorptions/H2--He/H2--He-NatAbund = H2--He-NatAbund__BoRi.DeltaWavenumber2_0.5-500mu.ciatable.petitRADTRANS.h5\n',
]

paths_idx = text.find('[Paths]')
if paths_idx != -1:
    text = text[:paths_idx] + ''.join(aliases) + text[paths_idx:]

config_path.write_text(text, encoding='utf-8')
PY

python - <<'PY'
from petitRADTRANS.radtrans import Radtrans
from proteus.utils.constants import prt_cia_species, prt_rayleigh_species

line_species = [
    'H2O',
    'H2',
    'CO2',
    'CO',
    'CH4',
    'SO2',
    'H2S',
    'O2',
    'NH3',
    'OH',
]

radtrans = Radtrans(
    line_species=line_species,
    rayleigh_species=prt_rayleigh_species,
    gas_continuum_contributors=prt_cia_species,
)
PY

cd $olddir

echo "Done!!"
