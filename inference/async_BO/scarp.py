from inference.async_BO.objective import run_proteus

par =  {'struct.mass_tot': 1.731102219365829, 'struct.corefrac': 0.352695195292852, 'atmos_clim.surf_greyalbedo': 0.2580284654470691, 'atmos_clim.dummy.gamma': 0.1139231975710947, 'escape.dummy.rate': 38166.62103263826, 'interior.dummy.ini_tmagma': 2923.5766546203363, 'outgas.fO2_shift_IW': 2.0355567442664615, 'delivery.radio_K': 82.28150669930466, 'delivery.elements.H_oceans': 5.021989174284928}

run_proteus(parameters=par, worker=0, iter=0)
