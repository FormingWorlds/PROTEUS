from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._config import Config


def _raise(exception: BaseException, msg: str):
    raise exception(msg)  # type: ignore


def _convert_spectral_file(config: Config) -> str:
    obj = getattr(config.atmos_clim, config.atmos_clim.module)

    bands = getattr(obj, 'spectral_bands')
    group = getattr(obj, 'spectral_group')

    return f'spectral_files/{group}/{bands}/{group}.sf'


def _atmosphere_nlev(config: Config) -> str:
    obj = getattr(config.atmos_clim, config.atmos_clim.module)
    return obj.num_levels


def _p_top(config: Config) -> float:
    obj = getattr(config.atmos_clim, config.atmos_clim.module)
    return obj.p_top


def _stellar_heating(config: Config):
    raise ValueError('`stellar_heating` has been deprecated')


def _shallow_ocean_layer(config: Config):
    raise ValueError("Handled by `atmos_clim.surf_state = 'mixed_layer'`")


def _solvevol_use_params(config: Config):
    raise ValueError("Equivalent to `delivery.initial = 'elements'`")


def _phi_global(config: Config):
    raise ValueError('No longer used')


COMPAT_MAPPING = {
    'star_model': ('star', 'mors', 'tracks'),
    'star_mass': ('star', 'mass'),
    'star_luminosity_modern': ('star', 'lum_now'),
    'star_age_modern': ('star', 'age_now'),
    'star_rot_pctle': ('star', 'rot_pctle'),
    'star_spectrum': ('star', 'mors', 'spec'),
    'semimajoraxis': ('orbit', 'semimajoraxis'),
    'mass': ('struct', 'mass'),
    'radius': ('struct', 'radius'),
    'zenith_angle': ('orbit', 'zenith_angle'),
    'asf_scalefactor': ('orbit', 's0_factor'),
    'albedo_s': ('atmos_clim', 'surf_albedo'),
    'albedo_pl': ('atmos_clim', 'albedo_pl'),
    'eccentricity': ('orbit', 'eccentricity'),
    'P_top': _p_top,
    'iter_max': ('params', 'stop', 'iters', 'maximum'),
    'log_level': ('params', 'out', 'logging'),
    'dir_output': ('params', 'out', 'path'),
    'time_star': ('star', 'age_ini'),
    'time_target': ('params', 'stop', 'time', 'maximum'),
    'spectral_file': _convert_spectral_file,
    'stellar_heating': _stellar_heating,
    'plot_iterfreq': ('params', 'out', 'plot_mod'),
    'plot_format': ('params', 'out', 'plot_fmt'),
    'sspec_dt_update': ('params', 'dt', 'starspec'),
    'sinst_dt_update': ('params', 'dt', 'starinst'),
    'dt_maximum': ('params', 'dt', 'maximum'),
    'dt_minimum': ('params', 'dt', 'minimum'),
    'dt_method': ('params', 'dt', 'method'),
    'dt_propconst': ('params', 'dt', 'proportional', 'propconst'),
    'dt_atol': ('params', 'dt', 'adaptive', 'atol'),
    'dt_rtol': ('params', 'dt', 'adaptive', 'rtol'),
    'dt_initial': ('params', 'dt', 'initial'),
    'shallow_ocean_layer': _shallow_ocean_layer,
    'F_atm_bc': ('atmos_clim', 'janus', 'F_atm_bc'),
    'skin_d': ('atmos_clim', 'surface_d'),
    'skin_k': ('atmos_clim', 'surface_k'),
    'prevent_warming': ('atmos_clim', 'prevent_warming'),
    'solid_stop': ('params', 'stop', 'solid', 'enabled'),
    'phi_crit': ('params', 'stop', 'solid', 'phi_crit'),
    'steady_stop': ('params', 'stop', 'steady', 'enabled'),
    'steady_flux': ('params', 'stop', 'steady', 'F_crit'),
    'steady_dprel': ('params', 'stop', 'steady', 'dprel'),
    'emit_stop': ('params', 'stop', 'radeqm', 'enabled'),
    'F_crit': ('params', 'stop', 'radeqm', 'F_crit'),
    'escape_model': ('escape', 'module'),
    'escape_stop': ('params', 'stop', 'escape', 'mass_frac'),
    'escape_dummy_rate': ('escape', 'dummy', 'rate'),
    'atmosphere_model': ('atmos_clim', 'module'),
    'atmosphere_surf_state': ('atmos_clim', 'surf_state'),
    'atmosphere_nlev': _atmosphere_nlev,
    'min_temperature': ('atmos_clim', 'tmp_minimum'),
    'max_temperature': ('atmos_clim', 'tmp_maximum'),
    'water_cloud': ('atmos_clim', 'cloud_enabled'),
    'alpha_cloud': ('atmos_clim', 'cloud_alpha'),
    'tropopause': ('atmos_clim', 'janus', 'tropopause'),
    'rayleigh': ('atmos_clim', 'rayleigh'),
    'atmosphere_chemistry': ('atmos_clim', 'agni', 'chemistry'),
    'interior_model': ('interior', 'module'),
    'interior_nlev': ('interior', 'spider', 'num_levels'),
    'grain_size': ('interior', 'grain_size'),
    'mixing_length': ('interior', 'spider', 'mixing_length'),
    'solver_tolerance': ('interior', 'spider', 'tolerance'),
    'tsurf_poststep_change': ('interior', 'spider', 'tsurf_atol'),
    'tsurf_poststep_change_frac': ('interior', 'spider', 'tsurf_rtol'),
    'planet_coresize': ('struct', 'corefrac'),
    'ic_adiabat_entropy': ('interior', 'spider', 'ini_entropy'),
    'ic_dsdr': ('interior', 'spider', 'ini_dsdr'),
    'F_atm': ('interior', 'F_initial'),
    'fO2_shift_IW': ('outgas', 'fO2_shift_IW'),
    'solvevol_use_params': _solvevol_use_params,
    'Phi_global': _phi_global,
    'CH_ratio': ('delivery', 'elements', 'CH_ratio'),
    'hydrogen_earth_oceans': ('delivery', 'elements', 'H_oceans'),
    'nitrogen_ppmw': ('delivery', 'elements', 'N_ppmw'),
    'sulfur_ppmw': ('delivery', 'elements', 'S_ppmw'),
    'H2O_included': ('outgas', 'calliope', 'include_H2O'),
    'H2O_initial_bar': ('delivery', 'volatiles', 'H2O'),
    'CO2_included': ('outgas', 'calliope', 'include_CO2'),
    'CO2_initial_bar': ('delivery', 'volatiles', 'CO2'),
    'N2_included': ('outgas', 'calliope', 'include_N2'),
    'N2_initial_bar': ('delivery', 'volatiles', 'N2'),
    'S2_included': ('outgas', 'calliope', 'include_S2'),
    'S2_initial_bar': ('delivery', 'volatiles', 'S2'),
    'SO2_included': ('outgas', 'calliope', 'include_SO2'),
    'SO2_initial_bar': ('delivery', 'volatiles', 'SO2'),
    'H2_included': ('outgas', 'calliope', 'include_H2'),
    'H2_initial_bar': ('delivery', 'volatiles', 'H2'),
    'CH4_included': ('outgas', 'calliope', 'include_CH4'),
    'CH4_initial_bar': ('delivery', 'volatiles', 'CH4'),
    'CO_included': ('outgas', 'calliope', 'include_CO'),
    'CO_initial_bar': ('delivery', 'volatiles', 'CO'),
}
