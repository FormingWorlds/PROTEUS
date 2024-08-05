# Usage

## Running PROTEUS

You can directly run PROTEUS using the Python command:

> ``` console
> $  python proteus.py --cfg [cfgfile]
> ```

Where `[cfgfile]` is the path to the required configuration file. If
`--cfg [cfgfile]` is not provided, then the default configuration
located at `input/default.cfg` will be used. Pass the flag `--resume` in
order to resume the simulation from the disk.

You can also run PROTEUS inside a Screen session using:

> ``` console
> $  tools/RunPROTEUS.sh [cfgfile] [alias] [resume] [detach]
> ```

Which runs PROTEUS using the config file `[cfgfile]` inside a Screen
session with the name `[alias]`. The `[resume]` parameter (y/n) tells
the model whether to resume from a previous state. The `[detach]`
parameter (y/n) tells the session whether to immediately detach or not.
This allows multiple instances of the model to be dispatched easily and
safely.

## Configuration file

PROTEUS accepts config files containing parameters in the format
`key = value`. All of the parameters required to run the model are
listed below with short explanations of their purpose and the values
they accept. Configuration files can contain blank lines. Comments are
indicated with a \# symbol. A lot of these parameters are not validated
by the code, so it will misbehave if unreasonable inputs are provided.
Not all of these parameters will be used, depending on the
configuration, but they must all be passed via the config file.

  ----------------------------------------------------------------------------------------------------------------
  Name                           Description                   Deprecated   Type      Domain
  ------------------------------ ----------------------------- ------------ --------- ----------------------------
  `star_model`                   Evolution model to use for    False        Integer   0: Spada, 1: Baraffe.
                                 star                                                 

  `star_rot_percentile`          The percentile used to find   False        Float     0 to 100.
                                 rotation rate of star from a                         
                                 distribution when the Mors                           
                                 evolution model is selected.                         

  `star_mass`                    Mass of the host star, in     False        Float     Valid range depends on the
                                 units of solar masses.                               stellar model used. For the
                                                                                      Mors model, it should be
                                                                                      between 0.1 and 1.25 solar
                                                                                      masses. Values outside of
                                                                                      the valid range will be
                                                                                      clipped.

  `star_radius_modern`           Assumed radius of the host    False        Float     Greater than zero.
                                 star as observed today, in                           
                                 units of solar radii.                                

  `star_luminosity_modern`       Assumed luminosity of the     False        Float     Greater than zero.
                                 host star as observed today,                         
                                 in units of solar                                    
                                 luminosities.                                        

  `star_temperature_modern`      Assumed temperature of the    False        Float     Greater than zero.
                                 host star as observed today,                         
                                 in units of kelvin.                                  

  `star_age_modern`              Estimated age of the host     False        Float     Greater than zero. Values
                                 star as observed today, in                           outside of the valid range
                                 units of years.                                      will be clipped.

  `star_rot_pctle`               Rotation rate percentile for  False        Float     Between 0 and 100.
                                 the star, relative to other                          
                                 stars of the same mass.                              

  `star_spectrum`                The spectrum of the host star False        String    Path to file, measured
                                 as observed today. These                             relative to the FWL_DATA
                                 files may be obtained using                          directory.
                                 the `GetStellarSpectrum`                             
                                 tool.                                                

  `mean_distance`                Distance between the planet   False        Float     Greater than zero.
                                 and its host star, in units                          
                                 of AU.                                               

  `mass`                         Mass of the planet, in units  False        Float     Greater than zero.
                                 of Earth mass.                                       

  `radius`                       Radius of the planet at the   False        Float     Greater than zero.
                                 surface, in units of Earth                           
                                 radius.                                              

  `zenith_angle`                 Angle of the incoming stellar False        Float     Positive values less than 90
                                 radiation relative to the                            degrees.
                                 zenith, in units of degrees.                         

  `asf_scalefactor`              Scale factor for the absorbed False        Float     Greater than zero.
                                 stellar flux (ASF), used in                          
                                 combination with                                     
                                 `zenith_angle`; see Cronin+14                        
                                 for a discussion on this.                            

  `albedo_s`                     Albedo of the surface of the  False        Float     Between zero and unity,
                                 planet.                                              inclusive.

  `albedo_pl`                    Bond albedo of the planet.    False        Float     Between zero and unity,
                                                                                      inclusive.

  `P_top`                        Pressure at the top of the    False        Float     Any reasonable positive
                                 atmosphere, in units of bar.                         value; 1e-5 works well.

  `dir_output`                   Name of the directory which   False        String    Name for a new folder to be
                                 will store the model output                          created inside the `output/`
                                 files. This includes data,                           folder.
                                 plots, temporary files, and                          
                                 config information.                                  

  `time_star`                    Age of the star at the start  False        Float     Greater than zero. Values
                                 of the simulation, in units                          outside of the valid range
                                 of years.                                            will be clipped.

  `time_target`                  Simulation time at which to   False        Float     Greater than zero.
                                 stop the model, if it hasn\'t                        
                                 stopped already, in units of                         
                                 years.                                               

  `spectral_file`                Spectral file to use when     False        String    Path to file measured
                                 running SOCRATES.                                    relative to the FWL_DATA
                                                                                      directory.

  `stellar_heating`              Flag to toggle stellar        False        Integer   0: disabled, 1: enabled
                                 heating, including the                               
                                 downward shortwave stream.                           

  `plot_iterfreq`                Iteration frequency at which  False        Integer   0: Do not make plots until
                                 to make (or update) the                              the simulation is complete.
                                 plots. Plots can be generated                        Values greater than 0: make
                                 during the simulation to                             plots every `plot_iterfreq`
                                 follow its progress and                              iterations.
                                 status.                                              

  `sspec_dt_update`              Time period at which to       False        Float     Greater than or equal to
                                 update the stellar spectrum                          zero.
                                 using the stellar evolution                          
                                 model of choice, in units of                         
                                 years.                                               

  `sinst_dt_update`              Period at which to update the False        Float     Greater than or equal to
                                 instellation flux and the                            zero.
                                 stellar radius using the                             
                                 stellar evolution model of                           
                                 choice, in units of years.                           

  `dt_maximum`                   Maximum allowable time-step   False        Float     Greater than zero.
                                 for the model, in units of                           
                                 years.                                               

  `dt_minimum`                   Minimum allowable time-step   False        Float     Greater than zero.
                                 for the model once the                               
                                 start-up phase has completed.                        
                                 Units of years.                                      

  `dt_method`                    Method to be used for         False        Integer   0: Proportional, 1:
                                 calculating the time-step                            Adaptive, 2: Maximum.
                                 once the start-up phase has                          
                                 completed. Units of years.                           
                                 \'Proportional\' sets `dt` to                        
                                 be some small fraction of the                        
                                 simulation time.                                     
                                 \'Adapative\' dynamically                            
                                 adjusts `dt` according to how                        
                                 rapidly the upward energy                            
                                 fluxes are changing.                                 
                                 \'Maximum\' sets `dt` to                             
                                 always be equal to                                   
                                 `dt_maximum`.                                        

  `dt_propconst`                 Proportionality constant when False        Float     Greater than zero.
                                 using `dt_method=0`. Time                            
                                 step is set by                                       
                                 `dt = t/dt_propconst`, so                            
                                 larger values mean smaller                           
                                 steps.                                               

  `dt_atol`                      Absolute tolerance on change  False        Float     Greater than zero.
                                 in flux and melt fraction for                        
                                 each iteration.                                      

  `dt_rtol`                      Relative tolerance on change  False        Float     Greater than zero.
                                 in flux and melt fraction for                        
                                 each iteration.                                      

  `dt_initial`                   Intial step size when using   False        Float     Greater than zero.
                                 `dt_method=1`, years.                                

  `shallow_ocean_layer`          Legacy method for converging  True         Integer   0: Off, 1: On
                                 atmospheric and interior                             
                                 upward fluxes.                                       

  `F_atm_bc`                     Boundary condition to use for False        Integer   0: Top of atmosphere, 1:
                                 calculating                                          Bottom of atmosphere.
                                 [F_atm]{.title-ref}. Can be                          
                                 set to either the top of the                         
                                 atmosphere or the bottom.                            

  `F_crit`                       Critical flux. Once the       False        Float     Greater than or equal to 0.
                                 upward net flux at the top of                        Set to 0 to disable.
                                 the atmosphere drops below                           
                                 this value, a smaller                                
                                 time-step is imposed.                                

  `escape_model`                 Escape model to be used.      False        Integer   0: None, 1: ZEPHYRUS, 2:
                                                                                      Dummy

  `escape_stop`                  Stop the simulation when the  False        Float     Values between zero and
                                 atmosphere mass drops below                          unity (exclusive).
                                 this fraction of its initial                         
                                 mass.                                                

  `escape_dummy_rate`            Bulk escape rate for dummy    False        Float     Any reasonable positive
                                 escape model \[kg s-1\]                              value.

  `prevent_warming`              Flag to ensure that the net   False        Integer   0: Disabled, 1: Enabled.
                                 upward energy flux is always                         
                                 positive, which prevents the                         
                                 star from causing net heating                        
                                 inside the planet.                                   

  `atmosphere_model`             Atmosphere model used to set  False        Integer   0: JANUS, 1: AGNI, 2: Dummy
                                 T(p) and T_surf.                                     

  `atmosphere_surf_state`        Surface boundary condition;   False        Integer   0: Free, 1: Fixed, 2:
                                 e.g. T_surf set by conductive                        Conductive.
                                 heat transport.                                      

  `skin_d`                       Conductive skin thickness,    False        Float     Greater than zero, metres.
                                 parameterising a thin layer                          
                                 at the surface.                                      

  `skin_k`                       Conductive skin thermal       False        Float     Greater than zero, \[W m-1
                                 conductivity.                                        K-1\].

  `atmosphere_nlev`              Number of atmosphere model    False        Integer   Greater than 15.
                                 levels, measured at                                  
                                 cell-centres.                                        

  `solid_stop`                   Flag to toggle the            False        Integer   0: Disabled, 1: Enabled.
                                 solidification break                                 
                                 condition.                                           

  `phi_crit`                     Value used for solidification False        Float     Values between zero and
                                 break condition; stop the                            unity.
                                 model once the global melt                           
                                 fraction drops below this                            
                                 value. This indiciates that                          
                                 the planet has solidified.                           
                                 Only applies when                                    
                                 `solid_stop` is enabled.                             

  `steady_stop`                  Flag to toggle the            False        Integer   0: Disabled, 1: Enabled.
                                 steady-state break condition.                        

  `steady_flux`                  Steady-state break condition, False        Float     Values between zero and
                                 requiring that                                       unity.
                                 `F_atm < steady_flux`.                               

  `steady_dprel`                 Steady-state break condition, False        Float     Values between zero and
                                 requiring that                                       unity.
                                 `dphi/dt < steady_dprel`.                            

  `min_temperature`              Temperature floor. The        False        Float     Greater than 0.
                                 temperature of the atmosphere                        
                                 is prevented from dropping                           
                                 below this value. Units of                           
                                 kelvin.                                              

  `max_temperature`              Temperature ceiling. The      False        Float     Greater than
                                 temperature of the atmosphere                        `min_temperature`.
                                 is prevented from reaching                           
                                 above this value. Units of                           
                                 kelvin.                                              

  `tropopause`                   Model of tropopause to be     False        Integer   0: None, 1: Skin, 2: Flux.
                                 used before, or in the                               
                                 absence of, a time-stepped                           
                                 solution to the temperature                          
                                 structure. \'None\' means no                         
                                 tropopause is applied.                               
                                 \'Skin\' means that the                              
                                 tropopause will be set to the                        
                                 radiative skin temperature.                          
                                 \'Flux\' dynamically sets the                        
                                 tropopause based on the                              
                                 heating rate.                                        

  `water_cloud`                  Enable water cloud radiative  False        Integer   0: Disabled, 1: Enabled.
                                 effects.                                             

  `alpha_cloud`                  Condensate retention          False        Float     Between 0 and 1, inclusive.
                                 fraction. A value of 0 means                         
                                 full rainout. A value of 1                           
                                 means full retention (cf.                            
                                 Li+2018).                                            

  `rayleigh`                     Enable rayleigh scattering.   False        Integer   0: Disabled, 1: Enabled.

  `atmosphere_chemistry`         Type of atmospheric chemistry False        Integer   0: None, 1: Equilibrium, 2:
                                 to apply at runtime. \'None\'                        Kinetics.
                                 applies no chemistry.                                
                                 \'Equilibrium\' uses                                 
                                 FastChem. \'Kinetics\' is not                        
                                 yet implemented.                                     

  `interior_nlev`                Number of levels used in the  False        Integer   Greater than 40.
                                 interior model                                       

  `grain_size`                   Size of crystal grains        False        Float     Any reasonable value greater
                                 considered within mushy                              than zero (for example, 0.1
                                 interior regions, units of                           metres)
                                 metres.                                              

  `mixing_length`                Mixing length                 False        Integer   1: Variable, 2: Constant.
                                 parameterisation to use in                           
                                 SPIDER. Can be constant or                           
                                 variable with depth.                                 

  `solver_tolerance`             Tolerance to provide to       False        Float     Greater than zero.
                                 SPIDER when it calls its                             
                                 numerical solver.                                    

  `tsurf_poststep_change`        Maximum allowed change in     False        Float     Greater than zero.
                                 surface temperature                                  
                                 calculated by SPIDER before                          
                                 it quits, to hand back to the                        
                                 other modules. Units of                              
                                 kelvin.                                              

  `tsurf_poststep_change_frac`   Maximum allowed relative      False        Float     Greater than zero.
                                 change in surface temperature                        
                                 calculated by SPIDER before                          
                                 it quits, to hand back to the                        
                                 other modules.                                       

  `planet_coresize`              Size of the planet\'s core as False        Float     Between zero and unity,
                                 a fraction of its total                              exclusive.
                                 interior radius.                                     

  `ic_adiabat_entropy`           Entropy at the surface for    False        Float     Greater than zero.
                                 intialising a SPIDER at the                          
                                 start of the run, in units of                        
                                 \[J kg-1 K-1\].                                      

  `ic_dsdr`                      Entropy gradient for          False        Float     Less than zero.
                                 intialising a SPIDER at the                          
                                 start of the run, in units of                        
                                 \[J kg-1 K-1 m-1\].                                  

  `F_atm`                        Initial guess for net upward  False        Float     Greater than zero.
                                 flux [F_atm]{.title-ref}.                            

  `fO2_shift_IW`                 Oxygen fugacity of the        False        Float     Any reasonable real value.
                                 interior, measured in log10                          
                                 units relative to the                                
                                 iron-wustite buffer. Positive                        
                                 values are oxidising,                                
                                 negative are reducing.                               

  `solvevol_use_params`          Flag to enable solving for    False        Integer   0: Disabled, 1: Enabled.
                                 initial partial pressures                            
                                 subject to interior                                  
                                 parameters, rather than using                        
                                 provided initial pressures.                          

  `Phi_global`                   Initial guess for mantle melt False        Float     Between 0 and 1, inclusive.
                                 fraction.                                            

  `CH_ratio`                     Required total-planet C/H     False        Float     Greater than zero.
                                 mass ratio. Used when                                
                                 `solvevol_use_params == 1`.                          

  `hydrogen_earth_oceans`        Total hydrogen inventory of   False        Float     Greater than zero.
                                 the planet. Used when when                           
                                 `solvevol_use_params == 1`.                          
                                 Units of Earth oceans                                
                                 equivalent.                                          

  `nitrogen_ppmw`                Nitrogen concentration. Used  False        Float     Greater than zero.
                                 when                                                 
                                 `solvevol_use_params == 1`.                          
                                 Parts per million of total                           
                                 mantle mass.                                         

  `sulfur_ppmw`                  Sulfur concentration. Used    False        Float     Greater than zero.
                                 when                                                 
                                 `solvevol_use_params == 1`.                          
                                 Parts per million of total                           
                                 mantle mass.                                         

  `X_included`                   Flag to include X in the      False        Integer   0: Excluded, 1: Included.
                                 model. For some (H2O, CO2,                           
                                 N2, S2) this will always                             
                                 equal 1.                                             

  `X_initial_bar`                Initial partial pressure of   False        Float     Greater than zero.
                                 X. Used when                                         
                                 `solvepp_enabled == 0`.                              
  ----------------------------------------------------------------------------------------------------------------

  : Model input parameters
