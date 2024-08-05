# Usage

## Running PROTEUS

You can directly run PROTEUS using the Python command:

``` console
python proteus.py --cfg [cfgfile]
```

Where `[cfgfile]` is the path to the required configuration file. If
`--cfg [cfgfile]` is not provided, then the default configuration
located at `input/default.cfg` will be used. Pass the flag `--resume` in
order to resume the simulation from the disk.

You can also run PROTEUS inside a Screen session using:

``` console
tools/RunPROTEUS.sh [cfgfile] [alias] [resume] [detach]
```

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

### Model input parameters

#### **star_model**

Evolution model to use for star

- Deprecated: False
- Type: Integer
- Domain: 0: Spada, 1: Baraffe

#### **star_rot_percentile**

The percentile used to find rotation rate of star from a distribution when the Mors evolution model is selected.  

- Deprecated: False
- Type: Float 
- Domain: 0 to 100.

#### **star_mass**

Mass of the host star, in units of solar masses.  

- Deprecated: False
- Type: Float 
- Domain: Valid range depends on the stellar model used. For the Mors model, it should be between 0.1 and 1.25 solar masses. Values outside of the valid range will be clipped.

#### **star_radius_modern**

Assumed radius of the host star as observed today, in units of solar radii.  

- Deprecated: False
- Type: Float
- Domain: Greater than zero.

#### **star_luminosity_modern**

Assumed luminosity of the host star as observed today, in units of solar luminosities.  

- Deprecated: False
- Type: Float
- Domain: Greater than zero.

#### **star_temperature_modern**

Assumed temperature of the host star as observed today, in units of kelvin.  

- Deprecated: False
- Type: Float
- Domain: Greater than zero.

#### **star_age_modern**

Estimated age of the host star as observed today, in units of years.  

- Deprecated: False
- Type: Float
- Domain: Greater than zero. Values outside of the valid range will be clipped.

#### **star_rot_pctle**

Rotation rate percentile for the star, relative to other stars of the same mass. 

- Deprecated: False
- Type: Float
- Domain: Between 0 and 100.

#### **star_spectrum**

The spectrum of the host star as observed today. These files may be obtained using the ``GetStellarSpectrum`` tool.  

- Deprecated: False
- Type: String
- Domain: Path to file, measured relative to the FWL_DATA directory.

#### **mean_distance**

Distance between the planet and its host star, in units of AU.  

- Deprecated: False
- Type: Float
- Domain: Greater than zero.

#### **mass**

Mass of the planet, in units of Earth mass.

- Deprecated: False
- Type: Float
- Domain: Greater than zero. 

#### **radius**

Radius of the planet at the surface, in units of Earth radius.  

- Deprecated: False
- Type: Float 
- Domain: Greater than zero.

#### **zenith_angle**

Angle of the incoming stellar radiation relative to the zenith, in units of degrees.    

- Deprecated: False
- Type: Float
- Domain: Positive values less than 90 degrees.

#### **asf_scalefactor**

Scale factor for the absorbed stellar flux (ASF), used in combination with ``zenith_angle``; see Cronin+14 for a discussion on this.    

- Deprecated: False
- Type: Float
- Domain: Greater than zero.

#### **albedo_s**

Albedo of the surface of the planet.    

- Deprecated: False
- Type: Float
- Domain: Between zero and unity, inclusive.

#### **albedo_pl**

Bond albedo of the planet.  

- Deprecated: False
- Type: Float
- Domain: Between zero and unity, inclusive.

#### **P_top**

Pressure at the top of the atmosphere, in units of bar.   

- Deprecated: False
- Type: Float
- Domain: Any reasonable positive value; 1e-5 works well.

#### **dir_output**

Name of the directory which will store the model output files. This includes data, plots, temporary files, and config information.  

- Deprecated: False
- Type: String
- Domain: Name for a new folder to be created inside the ``output/`` folder.

#### **time_star**

Age of the star at the start of the simulation, in units of years.   

- Deprecated: False
- Type: Float
- Domain: Greater than zero. Values outside of the valid range will be clipped.

#### **time_target**

Simulation time at which to stop the model, if it hasn't stopped already, in units of years.  

- Deprecated: False
- Type: Float
- Domain: Greater than zero.

#### **spectral_file**

Spectral file to use when running SOCRATES.   

- Deprecated: False
- Type: String
- Domain: Path to file measured relative to the FWL_DATA directory.

#### **stellar_heating**

Flag to toggle stellar heating, including the downward shortwave stream.  

- Deprecated: False
- Type: Integer
- Domain: 0: disabled, 1: enabled

#### **plot_iterfreq**

Iteration frequency at which to make (or update) the plots. Plots can be generated during the simulation to follow  its progress and status.   

- Deprecated: False
- Type: Integer
- Domain: 0: Do not make plots until the simulation is complete. Values greater than 0: make plots every ``plot_iterfreq`` iterations. 

#### **sspec_dt_update**

Time period at which to update the stellar spectrum using the stellar evolution model of choice, in units of years.   

- Deprecated: False
- Type: Float
- Domain: Greater than or equal to zero.

#### **sinst_dt_update**

Period at which to update the instellation flux and the stellar radius using the stellar evolution model of choice, in units of years.    

- Deprecated: False
- Type: Float
- Domain: Greater than or equal to zero.

#### **dt_maximum**

Maximum allowable time-step for the model, in units of years.    

- Deprecated: False
- Type: Float
- Domain: Greater than zero.

#### **dt_minimum**

Minimum allowable time-step for the model once the start-up phase has completed. Units of years.     

- Deprecated: False
- Type: Float
- Domain: Greater than zero.

#### **dt_method**

Method to be used for calculating the time-step once the start-up phase has completed. Units of years. 'Proportional' sets ``dt`` to be some small fraction of the simulation time. 'Adapative' dynamically adjusts ``dt`` according to how rapidly the upward energy fluxes are changing. 'Maximum' sets ``dt`` to always be equal to ``dt_maximum``.  

- Deprecated: False
- Type: Integer
- Domain: 0: Proportional, 1: Adaptive, 2: Maximum.    

#### **dt_propconst**

Proportionality constant when using ``dt_method=0``. Time step is set by ``dt = t/dt_propconst``, so larger values mean smaller steps.   

- Deprecated: False
- Type: Float
- Domain: Greater than zero.    

#### **dt_atol**

Absolute tolerance on change in flux and melt fraction for each iteration.   

- Deprecated: False
- Type: Float
- Domain: Greater than zero.    

#### **dt_rtol**

Relative tolerance on change in flux and melt fraction for each iteration.   

- Deprecated: False
- Type: Float
- Domain: Greater than zero.    

#### **dt_initial**

Intial step size when using ``dt_method=1``, years.

- Deprecated: False
- Type: Float
- Domain: Greater than zero.  

#### **shallow_ocean_layer**

Legacy method for converging atmospheric and interior upward fluxes. 

- Deprecated: True
- Type: Integer
- Domain: 0: Off, 1: On

#### **F_atm_bc**

Boundary condition to use for calculating `F_atm`. Can be set to either the top of the atmosphere or the bottom.     

- Deprecated: False
- Type: Integer
- Domain: 0: Top of atmosphere, 1: Bottom of atmosphere.    

#### **F_crit**

Critical flux. Once the upward net flux at the top of the atmosphere drops below this value, a smaller time-step is imposed.

- Deprecated: False
- Type: Float
- Domain: Greater than or equal to 0. Set to 0 to disable.    

#### **escape_model**

Escape model to be used.

- Deprecated: False
- Type: Integer
- Domain: 0: None, 1: ZEPHYRUS, 2: Dummy 

#### **escape_stop**

Stop the simulation when the atmosphere mass drops below this fraction of its initial mass.

- Deprecated: False
- Type: Float
- Domain: Values between zero and unity (exclusive).

#### **escape_dummy_rate**

Bulk escape rate for dummy escape model [kg s-1]

- Deprecated: False
- Type: Float
- Domain: Any reasonable positive value.

#### **prevent_warming**

Flag to ensure that the net upward energy flux is always positive, which prevents the star from causing net heating inside the planet.   

- Deprecated: False
- Type: Integer
- Domain: 0: Disabled, 1: Enabled.

#### **atmosphere_model**

Atmosphere model used to set T(p) and T_surf.    

- Deprecated: False
- Type: Integer
- Domain: 0: JANUS, 1: AGNI, 2: Dummy

#### **atmosphere_surf_state**

Surface boundary condition; e.g. T_surf set by conductive heat transport.   

- Deprecated: False
- Type: Integer
- Domain: 0: Free, 1: Fixed, 2: Conductive.

#### **skin_d``**

Conductive skin thickness, parameterising a thin layer at the surface.

- Deprecated: False
- Type: Float
- Domain: Greater than zero, metres.       

#### **skin_k``**

Conductive skin thermal conductivity.

- Deprecated: False
- Type: Float
- Domain: Greater than zero, [W m-1 K-1].    

#### **atmosphere_nlev**

Number of atmosphere model levels, measured at cell-centres.     

- Deprecated: False
- Type: Integer 
- Domain: Greater than 15.

#### **solid_stop**

Flag to toggle the solidification break condition.  

- Deprecated: False
- Type: Integer 
- Domain: 0: Disabled, 1: Enabled.

#### **phi_crit**

Value used for solidification break condition; stop the model once the global melt fraction drops below this value. This indiciates that the planet has solidified. Only applies when ``solid_stop`` is enabled.       

- Deprecated: False
- Type: Float
- Domain: Values between zero and unity.    

#### **steady_stop**

Flag to toggle the steady-state break condition.  

- Deprecated: False
- Type: Integer
- Domain: 0: Disabled, 1: Enabled.

#### **steady_flux**

Steady-state break condition, requiring that ``F_atm < steady_flux``.    

- Deprecated: False
- Type: Float
- Domain: Values between zero and unity.    

#### **steady_dprel**

Steady-state break condition, requiring that ``dphi/dt < steady_dprel``.

- Deprecated: False
- Type: Float
- Domain: Values between zero and unity.  

#### **min_temperature**

Temperature floor. The temperature of the atmosphere is prevented from dropping below this value. Units of kelvin.    

- Deprecated: False
- Type: Float
- Domain: Greater than 0.   

#### **max_temperature**

Temperature ceiling. The temperature of the atmosphere is prevented from reaching above this value. Units of kelvin.  

- Deprecated: False
- Type: Float
- Domain: Greater than ``min_temperature``.  

#### **tropopause**

Model of tropopause to be used before, or in the absence of, a time-stepped solution to the temperature structure. 'None' means no tropopause is applied. 'Skin' means that the tropopause will be set to the radiative skin temperature.  'Flux' dynamically sets the tropopause based on the heating rate. 

- Deprecated: False
- Type: Integer
- Domain: 0: None, 1: Skin, 2: Flux.

#### **water_cloud**

Enable water cloud radiative effects.

- Deprecated: False
- Type: Integer
- Domain: 0: Disabled, 1: Enabled.

#### **alpha_cloud**

Condensate retention fraction. A value of 0 means full rainout. A value of 1 means full retention (cf. Li+2018).

- Deprecated: False
- Type: Float
- Domain: Between 0 and 1, inclusive.

#### **rayleigh**

Enable rayleigh scattering.

- Deprecated: False
- Type: Integer
- Domain: 0: Disabled, 1: Enabled.

#### **atmosphere_chemistry**

Type of atmospheric chemistry to apply at runtime. 'None' applies no chemistry. 'Equilibrium' uses FastChem. 'Kinetics' is not yet implemented.

- Deprecated: False
- Type: Integer
- Domain: 0: None, 1: Equilibrium, 2: Kinetics.

#### **interior_nlev**

Number of levels used in the interior model

- Deprecated: False
- Type: Integer
- Domain: Greater than 40.

#### **grain_size**

Size of crystal grains considered within mushy interior regions, units of metres.

- Deprecated: False
- Type: Float
- Domain: Any reasonable value greater than zero (for example, 0.1 metres)

#### **mixing_length**

Mixing length parameterisation to use in SPIDER. Can be constant or variable with depth.

- Deprecated: False
- Type: Integer
- Domain: 1: Variable, 2: Constant.

#### **solver_tolerance**

Tolerance to provide to SPIDER when it calls its numerical solver.  

- Deprecated: False
- Type: Float
- Domain: Greater than zero.

#### **tsurf_poststep_change**

Maximum allowed change in surface temperature calculated by SPIDER before it quits, to hand back to the other modules. Units of kelvin.   

- Deprecated: False
- Type: Float
- Domain: Greater than zero.

#### **tsurf_poststep_change_frac**

Maximum allowed relative change in surface temperature calculated by SPIDER before it quits, to hand back to the other modules.   

- Deprecated: False
- Type: Float
- Domain: Greater than zero.

#### **planet_coresize**

Size of the planet's core as a fraction of its total interior radius.   

- Deprecated: False
- Type: Float
- Domain: Between zero and unity, exclusive.  

#### **ic_adiabat_entropy**

Entropy at the surface for intialising a SPIDER at the start of the run, in units of  [J kg-1 K-1].

- Deprecated: False
- Type: Float
- Domain: Greater than zero.

#### **ic_dsdr**

Entropy gradient for intialising a SPIDER at the start of the run, in units of  [J kg-1 K-1 m-1].

- Deprecated: False
- Type: Float
- Domain: Less than zero.

#### **F_atm**

Initial guess for net upward flux `F_atm`.  

- Deprecated: False
- Type: Float
- Domain: Greater than zero.

#### **fO2_shift_IW**

Oxygen fugacity of the interior, measured in log10 units relative to the iron-wustite buffer. Positive values are oxidising, negative are reducing.   

- Deprecated: False
- Type: Float
- Domain: Any reasonable real value.

#### **solvevol_use_params**

Flag to enable solving for initial partial pressures subject to interior parameters, rather than using provided initial pressures. 

- Deprecated: False
- Type: Integer
- Domain: 0: Disabled, 1: Enabled.

#### **Phi_global**

Initial guess for mantle melt fraction.    

- Deprecated: False
- Type: Float
- Domain: Between 0 and 1, inclusive.

#### **CH_ratio**

Required total-planet C/H mass ratio. Used when ``solvevol_use_params == 1``.    

- Deprecated: False
- Type: Float
- Domain: Greater than zero.

#### **hydrogen_earth_oceans**

Total hydrogen inventory of the planet. Used when when ``solvevol_use_params == 1``. Units of Earth oceans equivalent.  

- Deprecated: False
- Type: Float
- Domain: Greater than zero.

#### **nitrogen_ppmw**

Nitrogen concentration. Used when ``solvevol_use_params == 1``. Parts per million of total mantle mass.  

- Deprecated: False
- Type: Float
- Domain: Greater than zero. 

#### **sulfur_ppmw**

Sulfur concentration. Used when ``solvevol_use_params == 1``. Parts per million of total mantle mass.  

- Deprecated: False
- Type: Float
- Domain: Greater than zero.

#### **X_included**

Flag to include X in the model. For some (H2O, CO2, N2, S2) this will always equal 1.

- Deprecated: False
- Type: Integer
- Domain: 0: Excluded, 1: Included.

#### **X_initial_bar**

Initial partial pressure of X. Used when ``solvepp_enabled == 0``.    

- Deprecated: False
- Type: Float
- Domain: Greater than zero.