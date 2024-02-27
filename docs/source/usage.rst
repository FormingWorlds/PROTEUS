Usage
=====

Running PROTEUS
------------------------
 
You can directly run PROTEUS using the Python command:

   .. code-block:: console

      $  python proteus.py

Default settings and variables are set in ``init_coupler.cfg``.

You can also run PROTEUS using:

   .. code-block:: console

         $  tools/RunPROTEUS.sh [cfgfile] [alias] [detach]
   
Which runs PROTEUS using the config file ``[cfgfile]`` inside a Screen session 
with the name ``[alias]``. The ``[detatch]`` parameter (y/n) tells the session 
whether to immediately detach or not. his allows multiple instances of the model 
to be dispatched easily, while preventing runs from having clashing names.   
  
Configuration file    
------------------------   
 
PROTEUS accepts config files containing parameters in the format ``key = value``.
All of the parameters required to run the model are listed below with short 
explanations of their purpose and the values they accept. Configuration files 
can contain blank lines. Comments are indicated with a # symbol. A lot of these 
parameters are not validated by the code, so it will misbehave if unreasonable
inputs are provided. Not all of these parameters will be used, depending on the 
configuration, but they must all be passed via the config file.

.. list-table:: Model input parameters
   :header-rows: 1

   * - Name
     - Description
     - Deprecated
     - Type 
     - Domain

   * - ``star_model``
     - The star model to use for calculating the spectrum and luminosity. Options are the legacy implementation, the Baraffe model, and the Mors model.  
     - False
     - Integer
     - 0: Legacy, 1: Mors, 2: Baraffe.

   * - ``star_rot_percentile``
     - The percentile used to find rotation rate of star from a distribution when the Mors evolution model is selected.  
     - False
     - Float 
     - 0 to 100.

   * - ``star_mass``
     - Mass of the host star, in units of solar masses.  
     - False
     - Float 
     - Valid range depends on the stellar model used. For the Mors model, it should be between 0.1 and 1.25 solar masses. Values outside of the valid range will be clipped.

   * - ``star_radius_modern``
     - Assumed radius of the host star as observed today, in units of solar radii.  
     - False
     - Float
     - Greater than zero.

   * - ``star_luminosity_modern``
     - Assumed luminosity of the host star as observed today, in units of solar luminosities.  
     - False
     - Float
     - Greater than zero.

   * - ``star_temperature_modern``
     - Assumed temperature of the host star as observed today, in units of kelvin.  
     - False
     - Float
     - Greater than zero.

   * - ``star_age_modern``
     - Estimated age of the host star as observed today, in units of years.  
     - False
     - Float
     - Greater than zero. Values outside of the valid range will be clipped.

   * - ``star_spectrum``
     - The spectrum of the host star as observed today. These files may be obtained using the ``GetStellarSpectrum`` tool.  
     - False
     - String
     - Path to file, measured relative to the PROTEUS base directory.

   * - ``star_btrack``
     - Baraffe evolutionary track to be used when ``star_model = 1``.  
     - False
     - String
     - Path to file, measured relative to the PROTEUS base directory.

   * - ``mean_distance``
     - Distance between the planet and its host star, in units of AU.  
     - False
     - Float
     - Greater than zero.

   * - ``mass``
     - Mass of the planet, in units of kg.  
     - False
     - Float
     - Greater than zero. 

   * - ``radius``
     - Radius of the planet at the surface, in units of m.  
     - False
     - Float 
     - Greater than zero.

   * - ``zenith_angle``
     - Angle of the incoming stellar radiation relative to the zenith, in units of degrees.    
     - False
     - Float
     - Positive values less than 90 degrees.

   * - ``asf_scalefactor``
     - Scale factor for the absorbed stellar flux (ASF), used in combination with ``zenith_angle``; see Cronin+14 for a discussion on this.    
     - False
     - Float
     - Greater than zero.

   * - ``albedo_s``
     - Albedo of the surface of the planet.    
     - False
     - Float
     - Between zero and unity, inclusive.

   * - ``albedo_pl``
     - Bond albedo of the planet.  
     - False
     - Float
     - Between zero and unity, inclusive.

   * - ``P_top``
     - Pressure at the top of the atmosphere, in units of bar.   
     - False
     - Float
     - Any reasonable positive value; 1e-5 works well.

   * - ``dir_output``
     - Name of the directory which will store the model output files. This includes data, plots, temporary files, and config information.  
     - False
     - String
     - Name for a new folder to be created inside the ``output/`` folder.

   * - ``time_star``
     - Age of the star at the start of the simulation, in units of years.   
     - False
     - Float
     - Greater than zero. Values outside of the valid range will be clipped.

   * - ``time_planet``
     - Age of the planet at the start of the simulation, in units of years.  
     - False
     - Float
     - Greater than zero.

   * - ``time_target``
     - Simulation time at which to stop the model, if it hasn't stopped already, in units of years.  
     - False
     - Float
     - Greater than ``time_planet``.

   * - ``spectral_file``
     - Spectral file to use when running SOCRATES.   
     - False
     - String
     - Path to file measured relative to the ``JANUS/`` folder.

   * - ``stellar_heating``
     - Flag to toggle stellar heating, including the downward shortwave stream.  
     - False
     - Integer
     - 0: disabled, 1: enabled

   * - ``plot_iterfreq``
     - Iteration frequency at which to make (or update) the plots. Plots can be generated during the simulation to follow  its progress and status.   
     - False
     - Integer
     - 0: Do not make plots until the simulation is complete. Values greater than 0: make plots every ``plot_iterfreq`` iterations. 

   * - ``sspec_dt_update``
     - Time period at which to update the stellar spectrum using the stellar evolution model of choice, in units of years.   
     - False
     - Float
     - Greater than or equal to zero.

   * - ``sinst_dt_update``
     - Period at which to update the instellation flux and the stellar radius using the stellar evolution model of choice, in units of years.    
     - False
     - Float
     - Greater than or equal to zero.

   * - ``dt_maximum``
     - Maximum allowable time-step for the model, in units of years.    
     - False
     - Float
     - Greater than zero.

   * - ``dt_minimum``
     - Minimum allowable time-step for the model once the start-up phase has completed. Units of years.     
     - False
     - Float
     - Greater than zero.

   * - ``dt_method``
     - Method to be used for calculating the time-step once the start-up phase has completed. Units of years. 'Proportional' sets ``dt`` to be some small fraction of the simulation time. 'Adapative' dynamically adjusts ``dt`` according to how rapidly the upward energy fluxes are changing. 'Maximum' sets ``dt`` to always be equal to ``dt_maximum``.  
     - False
     - Integer
     - 0: Proportional, 1: Adaptive, 2: Maximum.    

   * - ``dt_propconst``
     - Proportionality constant when using ``dt_method=0``. Time step is set by ``dt = t/dt_propconst``, so larger values mean smaller steps.   
     - False
     - Float
     - Greater than zero.    

   * - ``dt_atol``
     - Absolute tolerance on change in flux and melt fraction for each iteration.   
     - False
     - Float
     - Greater than zero.    

   * - ``dt_rtol``
     - Relative tolerance on change in flux and melt fraction for each iteration.   
     - False
     - Float
     - Greater than zero.    

   * - ``flux_convergence``
     - Method to be used for converging atmospheric and interior upward fluxes. 'Off' applies nothing special, and allows SPIDER to determine the surface temperature. 'Restart' uses a shallow mixed ocean layer with a given heat capacity to balance the fluxes and obtain a surface temperature. 'On' waits until certain conditions are met, and then applies the 'Restart' method. 
     - True
     - Integer
     - 0: Off, 1: On, 2: Restart.   

   * - ``F_atm_bc``
     - Boundary condition to use for calculating `F_atm`. Can be set to either the top of the atmosphere or the bottom.     
     - False
     - Integer
     - 0: Top of atmosphere, 1: Bottom of atmosphere.    

   * - ``F_crit``
     - Critical flux. Once the upward net flux at the top of the atmosphere drops below this value, a smaller time-step is imposed.
     - False
     - Float
     - Greater than or equal to 0. Set to 0 to disable.    

   * - ``RF_crit``
     - Deprecated.     
     - True
     - Float 
     - ??

   * - ``dTs_atm``
     - Deprecated.      
     - True
     - Float
     - ??

   * - ``prevent_warming``
     - Flag to ensure that the net upward energy flux is always positive, which prevents the star from causing net heating inside the planet.   
     - False
     - Integer
     - 0: Disabled, 1: Enabled.

   * - ``atmosphere_model``   
     - Atmosphere model used to set T(p) and T_surf.    
     - False
     - Integer
     - 0: JANUS, 1: AGNI, 2: Dummy

   * - ``atmosphere_solve_energy``   
     - Enable radiative-convective solution for calculating T(p). Only available with AGNI.
     - False
     - Integer
     - 0: Disabled, 1: Enabled

   * - ``atmosphere_surf_state``   
     - Surface boundary condition; e.g. T_surf set by conductive heat transport.   
     - False
     - Integer
     - 0: Free, 1: Fixed, 2: Conductive.

   * - ``skin_d``  
     - Conductive skin thickness, parameterising a thin layer at the surface.
     - False
     - Float
     - Greater than zero, metres.       

   * - ``skin_k``  
     - Conductive skin thermal conductivity.
     - False
     - Float
     - Greater than zero, [W m-1 K-1].    

   * - ``atmosphere_nlev``   
     - Number of atmosphere model levels, measured at cell-centres.     
     - False
     - Integer 
     - Greater than 15.

   * - ``solid_stop``
     - Flag to toggle the solidification break condition.  
     - False
     - Integer 
     - 0: Disabled, 1: Enabled.

   * - ``phi_crit``
     - Value used for solidification break condition; stop the model once the global melt fraction drops below this value. This indiciates that the planet has solidified. Only applies when ``solid_stop`` is enabled.       
     - False
     - Float
     - Values between zero and unity.    

   * - ``steady_stop``
     - Flag to toggle the steady-state break condition.  
     - False
     - Integer
     - 0: Disabled, 1: Enabled.

   * - ``steady_flux``
     - Steady-state break condition, requiring that ``F_atm < steady_flux``.    
     - False
     - Float
     - Values between zero and unity.    

   * - ``steady_dprel``
     - Steady-state break condition, requiring that ``dphi/dt < steady_dprel``.
     - False
     - Float
     - Values between zero and unity.  

   * - ``N2_partitioning``
     - The melt-vapour partitioning of the N2 volatile is redox-state dependent. Use this flag to determine which parameterisation will be calculated.   
     - False
     - Integer
     - 0: Oxidised, 1: Reduced.

   * - ``min_temperature``
     - Temperature floor. The temperature of the atmosphere is prevented from dropping below this value. Units of kelvin.    
     - False
     - Float
     - Greater than 0.   

   * - ``max_temperature``
     - Temperature ceiling. The temperature of the atmosphere is prevented from reaching above this value. Units of kelvin.  
     - False
     - Float
     - Greater than ``min_temperature``.  

   * - ``tropopause``
     - Model of tropopause to be used before, or in the absence of, a time-stepped solution to the temperature structure. 'None' means no tropopause is applied. 'Skin' means that the tropopause will be set to the radiative skin temperature.  'Flux' dynamically sets the tropopause based on the heating rate. 
     - False
     - Integer
     - 0: None, 1: Skin, 2: Flux.

   * - ``insert_rscatter``
     - Enable rayleigh scattering.
     - False
     - Integer
     - 0: Disabled, 1: Enabled.

   * - ``atmosphere_chem_type``
     - Type of atmospheric chemistry to apply with VULCAN. 'None' applies no chemistry. 'Offline' provides the files required for running it offline. 'Online' is not yet implemented.
     - False
     - Integer
     - 0: None, 1: Offline, 2: Online.

   * - ``IC_INTERIOR``
     - Initial condition for SPIDER's interior component. 'Fresh' begins the simulation using the conditions provided. 'Restart' tries to pick up from a previous run, but is currently unsupported    
     - False
     - Integer 
     - 1: Fresh, 2: Restart.

   * - ``SEPARATION``
     - Flag to include gravitational separation of solid/melt in SPIDER.     
     - False
     - Integer
     - 0: Disabled, 1: Enabled.

   * - ``mixing_length``
     - Mixing length parameterisation to use in SPIDER. Can be constant or variable with depth.
     - False
     - Integer
     - 1: Variable, 2: Constant.

   * - ``PARAM_UTBL``
     - Flag to include an ultra-thin thermal boundary layer (UTBL) in SPIDER. This is used to parameterise the under-resolved conductive layer at the surface. Not compatible with ``atmosphere_surf_state==2``.   
     - False
     - Integer
     - 0: Disabled, 1: Enabled.

   * - ``solver_tolerance``
     - Tolerance to provide to SPIDER when it calls its numerical solver.  
     - False
     - Float
     - Greater than zero.

   * - ``tsurf_poststep_change``
     - Maximum allowed change in surface temperature calculated by SPIDER before it quits, to hand back to the other modules. Units of kelvin.   
     - False
     - Float
     - Greater than zero.

   * - ``tsurf_poststep_change_frac``
     - Maximum allowed relative change in surface temperature calculated by SPIDER before it quits, to hand back to the other modules.   
     - False
     - Float
     - Greater than zero.

   * - ``planet_coresize``
     - Size of the planet's core as a fraction of its total interior radius.   
     - False
     - Float
     - Between zero and unity, exclusive.  

   * - ``ic_interior_filename``
     - Resume PROTEUS from this SPIDER JSON file. Currently untested.   
     - False
     - String 
     - Path to file.

   * - ``ic_adiabat_entropy``
     - Entropy at the surface for intialising a SPIDER at the start of the run, in units of  [J kg-1 K-1].
     - False
     - Float
     - Greater than zero.

   * - ``ic_dsdr``
     - Entropy gradient for intialising a SPIDER at the start of the run, in units of  [J kg-1 K-1 m-1].
     - False
     - Float
     - Less than zero.

   * - ``F_atm``
     - Initial guess for net upward flux `F_atm`. Your choice for this value will depend on where `F_atm` is measured (see ``F_atm_bc``).   
     - False
     - Float
     - Greater than zero.

   * - ``fO2_shift_IW``
     - Oxygen fugacity of the interior, measured in log10 units relative to the iron-wustite buffer. Positive values are oxidising, negative are reducing.   
     - False
     - Float
     - Any reasonable real value.

   * - ``solvepp_enabled``
     - Flag to enable solving for initial partial pressures subject to interior parameters, equilibrium reactions, and melt-vapour partitioning.   
     - False
     - Integer
     - 0: Disabled, 1: Enabled.

   * - ``T_surf_guess``
     - Initial guess for surface temperature when ``solvepp_enabled == 1``, units of kelvin.
     - False
     - Float
     - Greater than zero.
   
   * - ``melt_fraction_guess``
     - Initial guess for mantle melt fraction when ``solvepp_enabled == 1``.    
     - False
     - Float
     - Between 0 and 1, inclusive.

   * - ``CH_ratio``
     - Initial guess for C/H ratio when ``solvepp_enabled == 1``.    
     - False
     - Float
     - Greater than zero.

   * - ``hydrogen_earth_oceans``
     - Total hydrogen inventory of the planet, used when when ``solvepp_enabled == 1``.   Units of Earth oceans equivalent.  
     - False
     - Float
     - Greater than zero.

   * - ``nitrogen_ppmw``
     - Initial nitrogen concentration in the mantle when ``solvepp_enabled == 1``. Parts per million of total mantle mass.  
     - False
     - Float
     - Greater than zero. 

   * - ``X_included``
     - Flag to include or exclude a volatile X from SPIDER and SOCRATES. This value may be overwritten at runtime but will be saved to the helpfile.  
     - False
     - Integer
     - 0: Excluded, 1: Included.

   * - ``X_add_bars``
     - Bars of a volatile X to add to the system at the start of the model run. When ``solvepp_enabled == 1``, these bars are included in addition to those found by the partial pressure solver. Otherwise, this parameter is how you specify the initial volatile inventory of the planet.    
     - False
     - Float
     - Greater than zero.

   * - ``X_poststep_change``
     - Relative change in the abundance of a volatile X, for which SPIDER will exit early to pass back to other modules within PROTEUS. Prevents large single-step changes from occuring.     
     - False
     - Float
     - Greater than zero.


