#!/bin/bash
# Pack PROTEUS grid output into a single folder

oldpwd=$(pwd)

grid=$(realpath $1)
echo "Grid dir: $grid"

# packing dir
pack=$grid/pack
echo "Pack dir: $pack"
rm -rf $pack
mkdir $pack

sleep 3
cd $grid

# check subfolders exist
if compgen -G "./case_*/" > /dev/null; then
    echo "Case subfolders exist"
else
    echo "Cannot find any subfolders containing grid cases!"
    exit 1
fi

# copy toplevel files
cp manager.log $pack/
cp ref_config.toml $pack/
cp copy.grid.toml $pack/

# make copies of grid case files
echo "Copy results"
for dir in ./case_*/     # list directories in the form "/tmp/dirname/"
do
    dir=${dir%*/}      # remove the trailing "/"
    dir=$(basename $dir)
    num=$(echo $dir | cut -d '_' -f 2)

    echo "   $num"
    rm -rf $pack/$num
    mkdir $pack/$num

    cp $dir/runtime_helpfile.csv $pack/$num/data.csv
    cp $dir/init_coupler.toml $pack/$num/config.toml
    cp $dir/*.log $pack/$num/
    cp -r $dir/plots $pack/$num/
done

# compress
cd $oldpwd
echo "Make zip"
zip -r $grid/pack.zip $pack

echo "Done!"
