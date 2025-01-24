from utils import run_proteus

import numpy as np

# if you get the name wrong it will create a new parameter!
# par = {
#                     "struct.mass_tot": 1.0,
#                     "struct.corefrac": 0.55,

#                     "atmos_clim.surf_greyalbedo": 0.1,
#                     "atmos_clim.dummy.gamma": 0.2,

#                     "escape.dummy.rate": 2.0e4,

#                     "interior.dummy.magma": 3500,

#                     "outgas.fO2_shift_IW": 2,

#                     "delivery.radio_K": 310.0,
#                     "delivery.elements.H_oceans": 6.0}


par = { "struct.mass_tot": 2.1382795033572477,
        "struct.corefrac": 0.32958394089386567,

        "atmos_clim.surf_greyalbedo": 0.5003406839007034,
        "atmos_clim.dummy.gamma": 0.5725611762292598,

        "escape.dummy.rate": 55222.03436489329,

        "interior.dummy.magma": 3529.0900543527023,

        "outgas.fO2_shift_IW": -0.9705082463067676,

        "delivery.radio_K": 267.84079438731965,
        "delivery.elements.H_oceans": 8.643922254864316}

out = run_proteus({"struct.mass_tot": 0.5}, run_name = "simple_BO/tmp2")

print(out)
