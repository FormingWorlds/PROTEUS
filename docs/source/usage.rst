Usage
=====

Running PROTEUS
----------------

You can execute PROTEUS using:

   .. code-block:: console

      $  python proteus.py

Default settings and variables are set in ``init_coupler.cfg``.


Configuration file
----------------

PROTEUS accepts config files containing parameters in the format `key = value`.
Only certain paramers are defined, all of which are listed below with short 
explanations of their purpose and the values they accept. Configuration files 
can contain blank lines. Comments are indicated with a # symbol.

* `star_model`
   - The star model to use for calculating the spectrum and luminosity. Options
   are the legacy implementation, the Baraffe model, and the Mors model. 
   - (Integer) 0: Legacy, 1: Mors, 2: Baraffe.

* `star_rot_percentile`
   - The percentile used to find rotation rate of star from a distribution when
   the Mors evolution model is selected.
   - (Float) 0 to 100.

* `star_mass`
   - Mass of the host star, in units of solar masses.
   - (Float) Valid range depends on the stellar model used. For the Mors model, 
   it should be between 0.1 and 1.25 solar masses. Values outside of the valid
   range will be clipped.

* `star_radius_modern`
   - Assumed radius of the host star as observed today, in units of solar radii.
   - (Float) Greater than zero.

* `star_luminosity_modern`
   - Assumed luminosity of the host star as observed today, in units of solar 
   luminosities.
   - (Float) Greater than zero.

* `star_temperature_modern`
   - Assumed temperature of the host star as observed today, in units of kelvin.
   - (Float) Greater than zero.

* `star_age_modern`
   - Estimated age of the host star as observed today, in units of years.
   - (Float) Greater than zero. Values outside of the valid range will be
   clipped.

* `star_spectrum`
   - The spectrum of the host star as observed today. These files may be 
   obtained using the `GetStellarSpectrum` tool.
   - (String) Path to file, measured relative to the PROTEUS base directory.

* `star_btrack`
   - Baraffe evolutionary track to be used when `star_model = 1`.
   - (String) Path to file, measured relative to the PROTEUS base directory.

* `mean_distance`
   - Distance between the planet and its host star, in units of AU.
   - (Float) Greater than zero.

* `mass`
   - Mass of the planet, in units of kg.
   - (Float) Greater than zero.

* `radius`
   - Radius of the planet, in units of m.
   - (Float) Greater than zero.

* `zenith_angle`
   - Angle of the incoming stellar radiation relative to the zenith, in units of
   degrees.
   - (Float) Positive values less than 90 degrees.

* `albedo_s`
   - Albedo of the surface of the planet
   - (Float) Between zero and unity, inclusive.

* `albedo_pl`
   - Bond albedo of the planet.
   - (Float) Between zero and unity, inclusive.

* `P_top`
   - Pressure at the top of the atmosphere, in units of bar.
   - (Float) Any reasonable positive value; 1e-5 works well.

* `dir_output`
   - Name of the directory which will store the model output files. This
   includes data, plots, temporary files, and config information.
   - (String) Name for a new folder to be created inside the `output/` folder.

* `time_star`
   - Age of the star at the start of the simulation, in units of years.
   - (Float) Greater than zero. Values outside of the valid range will be
   clipped.

* `time_planet`
   - Age of the planet at the start of the simulation, in units of years.
   - (Float) Greater than zero.

* `time_target`
   - Simulation time at which to stop the model, if it hasn't stopped already, 
   in units of years.
   - (Float) Greater than `time_planet`.

* `spectral_file`
   - Spectral file to use when running SOCRATES. 
   - (String) Path to file measured relative to the `AEOLUS/` folder.

* `stellar_heating`
   - Flag to toggle stellar heating, including the downward shortwave stream.
   - (Integer) 0: disabled, 1: enabled






