#bash script to run multiple proteus runs one after other
MIXFILE=vertical_mix
orb_dist=("0.01" "0.1" "1")
fO2=("5" "-5" "1" "-1")
Hoceans=("4")
silicates=("true")
CHratio=("0.1")
input_file=input/run_silicates.toml

# make directories for each composition and make fastchem grids for all compositions that don't have grids yet

for ofug in "${fO2[@]}"; do
	for orbit in "${orb_dist[@]}"; do
		for Hocean in "${Hoceans[@]}"; do
			for sil in "${silicates[@]}"; do
				echo "silicates are: "$sil
				for CH in "${CHratio[@]}"; do
					python3 tools_leoni/edit_input_file.py $input_file $ofug $orbit $Hocean $CH $sil
					proteus start -c $input_file
				done
			done
		done
	done
done
