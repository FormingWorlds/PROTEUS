#bash script to run multiple proteus runs one after other
MIXFILE=vertical_mix
orb_dist=("0.01" "0.05" "0.1" "0.5" "1")
fO2=("-5" "-4" "-3" "-2" "-1" "1" "2" "3" "4" "5")
Hoceans=("1" "2" "3" "4")
silicates("true" "false")
CH_ratio=("1")
input_file=input/run_silicates.toml

# make directories for each composition and make fastchem grids for all compositions that don't have grids yet

for ofug in "${fO2[@]}"; do
	echo $ofug
	for orbit in "${orb_dist[@]}"; do
		echo $orbit
		for Hocean in "${Hoceans[@]}"; do
			echo $Hocean
			for sil in "${silicates[@]}"; do
				echo "silicates are: "$sil
			for CH in "${CHratio[@]}"; do
				echo $CH
				python3 /data3/leoni/PROTEUS/tools_leoni/edit_input_file.py ${input_file} ${ofug} ${orbit} ${Hocean} ${CH} ${sil}
				proteus start -c ${input_file}

	done
	done
	done
	done
	done
