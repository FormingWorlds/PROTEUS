from utils import run_proteus, update_toml

# the reference config file
dummy_path = "input/demos/dummy.toml"

# name of this new config file
run_name = "temp3"

# if you get the name wrong it will create a new parameter!
update_params = {   "params.out.path": run_name,

                    "struct.mass_tot": 1.0,
                    "struct.corefrac": 0.55,

                    "atmos_clim.surf_greyalbedo": 0.1,
                    "atmos_clim.dummy.gamma": 0.2,

                    "escape.dummy.rate": 2.0e4,

                    "interior.dummy.magma": 3500,

                    "outgas.fO2_shift_IW": 2,

                    "delivery.radio_K": 310.0,
                    "delivery.elements.H_oceans": 6.0}


config_path = "input/demos/" + run_name + ".toml"

update_toml(dummy_path, update_params, config_path)

out_path = "output/" + run_name + "/runtime_helpfile.csv"

out = run_proteus(config_path, out_path)

print(out)
