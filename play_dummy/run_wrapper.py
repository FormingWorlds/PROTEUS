from utils import run_proteus
import time

# if you get the name wrong it will create a new parameter!
# par = {
#                     "struct.mass_tot": 1.0,
#                     "struct.corefrac": 0.55,

#                     "atmos_clim.surf_greyalbedo": 0.1,
#                     "atmos_clim.dummy.gamma": 0.2,

#                     "escape.dummy.rate": 2.0e4,

#                     "interior.dummy.ini_tmagma": 3500,

#                     "outgas.fO2_shift_IW": 2,

#                     "delivery.radio_K": 310.0,
#                     "delivery.elements.H_oceans": 6.0}


par =  {'struct.mass_tot': 1.731102219365829, 'struct.corefrac': 0.352695195292852, 'atmos_clim.surf_greyalbedo': 0.2580284654470691, 'atmos_clim.dummy.gamma': 0.1139231975710947, 'escape.dummy.rate': 38166.62103263826, 'interior.dummy.ini_tmagma': 2923.5766546203363, 'outgas.fO2_shift_IW': 2.0355567442664615, 'delivery.radio_K': 82.28150669930466, 'delivery.elements.H_oceans': 5.021989174284928}
t0 = time.time()

out = run_proteus(par, run_name = "BO/tmp2")

t1 = time.time()

print(f"Time taken: {t1 - t0:.2f} seconds")
print(out)
