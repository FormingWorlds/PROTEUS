#!/bin/bash
# Download and install petitRADTRANS

echo "Installing petitRADTRANS into Python environment..."

# Installing meson first
python -m pip install numpy meson-python ninja

# Make room
workpath="prt/"
rm -rf $workpath

# Download
echo "Cloning from GitHub"
if [ "$use_ssh" = true ]; then
    uri="git@gitlab.com:mauricemolli/petitRADTRANS.git"
else
    uri="https://gitlab.com/mauricemolli/petitRADTRANS.git"
fi
echo "    $uri -> $workpath"
git clone "$uri" "$workpath"

# Change dir and install
olddir=$(pwd)
cd $workpath
python -m pip install -U -e . --no-build-isolation

sed -i '46c\    except Exception:' petitRADTRANS/__file_conversion.py
sed -i '30c\    except Exception:' petitRADTRANS/opacities/opacities.py

echo "Installing petitRADTRANS opacity tables..."
prt_input_data_path="$(pwd)/input_data"
PRT_INPUT_DATA_PATH="$prt_input_data_path"
rm -rf ~/.petitradtrans/petitradtrans_config_file.ini

python - <<'PY'
import os

from petitRADTRANS.radtrans import Radtrans
radtrans = Radtrans(line_species=['Ti_+'])
PY

sed -i "s|^prt_input_data_path = .*|prt_input_data_path = $(pwd)/input_data|" ~/.petitradtrans/petitradtrans_config_file.ini
sed -i '/^\[Paths\]/i\
opacities/lines/correlated_k/H2O/1H2-16O = 1H2-16O__HITEMP.R1000_0.1-250mu.ktable.petitRADTRANS.h5\
opacities/lines/correlated_k/TiO/48Ti-16O = 48Ti-16O__McKemmish.R1000_0.1-250mu.ktable.petitRADTRANS.h5\
opacities/lines/correlated_k/VO/51V-16O = 51V-16O__Plez.R1000_0.1-250mu.ktable.petitRADTRANS.h5\
opacities/lines/correlated_k/Na/23Na = 23Na__Allard.R1000_0.1-250mu.ktable.petitRADTRANS.h5\
opacities/lines/correlated_k/K/39K = 39K__Allard.R1000_0.1-250mu.ktable.petitRADTRANS.h5\
opacities/lines/correlated_k/CH4/12C-1H4 = 12C-1H4__HITEMP.R1000_0.1-250mu.ktable.petitRADTRANS.h5\
opacities/continuum/collision_induced_absorptions/H2--H2/H2--H2-NatAbund = H2--H2-NatAbund__BoRi.R831_0.6-250mu.ciatable.petitRADTRANS.h5\
opacities/continuum/collision_induced_absorptions/H2--He/H2--He-NatAbund = H2--He-NatAbund__BoRi.DeltaWavenumber2_0.5-500mu.ciatable.petitRADTRANS.h5\
' ~/.petitradtrans/petitradtrans_config_file.ini

python - <<'PY'
import os

from petitRADTRANS.radtrans import Radtrans

prt_gases = ['H2', 'H2O', 'CH4', 'CO', 'CO2', 'O', 'NH3', 'O2', 'O3', 'H2S', 'HCN', 'OH', 'PH3', 'SiO', 'SO2', 'TiO', 'VO', 'Na', 'K', 'Ti', 'Fe', 'C2H2', 'FeH']
prt_rayleigh_species = ['H2', 'He', 'H']
prt_cia_species = ['H2--H2', 'H2--He', 'CO2--CO2', 'N2--H2', 'N2--He', 'N2--N2', 'N2--O2', 'O2--O2']

radtrans = Radtrans(line_species=prt_gases, rayleigh_species=prt_rayleigh_species, gas_continuum_contributors=prt_cia_species)
PY

cd $olddir

echo "Done!!"
