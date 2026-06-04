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
sed -i "s|^prt_input_data_path = .*|prt_input_data_path = $(pwd)/input_data|" ~/.petitradtrans/petitradtrans_config_file.ini
cd $olddir

echo "Done!!"
