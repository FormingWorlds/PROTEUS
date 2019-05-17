#!/usr/bin/env python

# import logging
import spider_utils as su
# import matplotlib.transforms as transforms
# import matplotlib.pyplot as plt
import numpy as np
# import os
# import matplotlib.ticker as ticker

# logger = su.get_my_logger(__name__)

#====================================================================
def write_surface_temp():

    # logger.info( 'building atmosphere' )

    sim_times = su.get_all_output_times()  # yr

    keys_t = ( ('atmosphere','mass_liquid'),
               ('atmosphere','mass_solid'),
               ('atmosphere','mass_mantle'),
               ('atmosphere','CO2','liquid_kg'),
               ('atmosphere','CO2','solid_kg'),
               ('atmosphere','CO2','initial_kg'),
               ('atmosphere','CO2','atmosphere_kg'),
               ('atmosphere','CO2','atmosphere_bar'),
               ('atmosphere','H2O','liquid_kg'),
               ('atmosphere','H2O','solid_kg'),
               ('atmosphere','H2O','initial_kg'),
               ('atmosphere','H2O','atmosphere_kg'),
               ('atmosphere','H2O','atmosphere_bar'),
               ('atmosphere','temperature_surface'),
               ('atmosphere','emissivity'),
               ('rheological_front_phi','phi_global'),
               ('atmosphere','Fatm'))

    data_a = su.get_dict_surface_values_for_times( keys_t, sim_times )

    mass_liquid_a = data_a[0,:]
    mass_solid_a = data_a[1,:]
    mass_mantle_a = data_a[2,:]
    mass_mantle = mass_mantle_a[0]          # time independent

    # compute total mass (kg) in each reservoir
    CO2_liquid_kg_a = data_a[3,:]
    CO2_solid_kg_a = data_a[4,:]
    CO2_total_kg_a = data_a[5,:]
    CO2_total_kg = CO2_total_kg_a[0]        # time-independent
    CO2_atmos_kg_a = data_a[6,:]
    CO2_atmos_a = data_a[7,:]
    CO2_escape_kg_a = CO2_total_kg - CO2_liquid_kg_a - CO2_solid_kg_a - CO2_atmos_kg_a

    H2O_liquid_kg_a = data_a[8,:]
    H2O_solid_kg_a = data_a[9,:]
    H2O_total_kg_a = data_a[10,:]
    H2O_total_kg = H2O_total_kg_a[0]        # time-independent
    H2O_atmos_kg_a = data_a[11,:]
    H2O_atmos_a = data_a[12,:]
    H2O_escape_kg_a = H2O_total_kg - H2O_liquid_kg_a - H2O_solid_kg_a - H2O_atmos_kg_a

    temperature_surface_a = data_a[13,:]
    emissivity_a = data_a[14,:]             # internally computed emissivity
    phi_global = data_a[15,:]               # global melt fraction
    Fatm = data_a[16,:]

    # output surface
    out_a = np.column_stack( (sim_times, temperature_surface_a ) )
    np.savetxt( 'surfaceT.dat', out_a )

#
# #====================================================================
# def main():
#
#     write_surface_temp()
#     # plt.show()
#
# #====================================================================
#
# if __name__ == "__main__":
#
#     main()
