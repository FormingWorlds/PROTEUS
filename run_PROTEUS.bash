#bash script to run multiple proteus runs one after other
orb_dist=("0.016" "0.064" "0.032")
fO2=("-1" "5" "-5")
mass=("9.0" "2.0")
silicates=("true")
Cabund=("1090.0" "109.0" "3270.0")
input_file=input/run_silicates.toml

# make directories for each composition and make fastchem grids for all compositions that don't have grids yet
for Cppmw in "${Cabund[@]}"; do
	for ma in "${mass[@]}"; do
		for ofug in "${fO2[@]}"; do
			for orbit in "${orb_dist[@]}"; do
				for sil in "${silicates[@]}"; do
					echo "silicates are: "$sil
					python3 tools_leoni/edit_input_file.py $input_file $ofug $orbit $ma $Cppmw $sil
					proteus start -c $input_file
				done
			done
		done
	done
done
