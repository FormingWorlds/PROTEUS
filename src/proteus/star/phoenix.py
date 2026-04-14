# Everything that has to do with PHOENIX stellar spectra
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from proteus.utils.constants import AU, M_sun, R_sun, const_G
from proteus.utils.data import GetFWLData, download_phoenix
from proteus.utils.helper import UpdateStatusfile
from proteus.utils.phoenix_helper import phoenix_filename, phoenix_param, phoenix_to_grid

log = logging.getLogger('fwl.' + __name__)

if TYPE_CHECKING:
    from proteus import Proteus

# Mass limits on stellar tracks [Msun]
MASS_LIM = {'spada': (0.10, 1.25), 'baraffe': (0.01, 1.40)}


def phoenix_params(handler: Proteus, stellar_track=None, age_yr: float | None = None):
    """
    Build PHOENIX parameters.

    Parameters
    ----------
    handler : Proteus
        Proteus object instance
    stellar_track :
        MORS stellar track object (mors.Star or BaraffeTrack), if available.
        If provided, it is used to compute Teff and radius when missing.
    age_yr : float, optional
        Stellar age in years at which PHOENIX should represent the star.
        If None, defaults to config.star.mors.age_now.

    Returns
    -------
    dict
        Dictionary with keys: 'Teff', 'logg', 'radius', 'FeH', 'alpha'.

        - Teff   : effective temperature [K]
        - logg   : surface gravity log10(g [cgs])
        - radius : stellar radius [R_sun]
        - FeH    : [Fe/H]
        - alpha  : [alpha/M]
    """
    star_cfg = handler.config.star
    mors_cfg = star_cfg.mors

    if age_yr is None:
        age_yr = mors_cfg.age_now * 1e9  # Gyr -> yr

    # Composition parameters from config
    FeH = mors_cfg.phoenix_FeH
    alpha = mors_cfg.phoenix_alpha

    if FeH == 0.0:
        log.info(
            'PHOENIX: Using solar metallicity [Fe/H]=0.0. Set star.mors.FeH to change the composition.'
        )
    if alpha == 0.0:
        log.info(
            'PHOENIX: Using solar [alpha/M]=0.0. Set star.mors.alpha to change the alpha fraction.'
        )

    # Overrides from config
    Teff = getattr(mors_cfg, 'phoenix_Teff', None)
    radius = getattr(mors_cfg, 'phoenix_radius', None)  # [R_sun]
    logg = getattr(mors_cfg, 'phoenix_log_g', None)  # log10(g [cgs])

    # If we have a stellar track, use it to fill Teff / radius
    if stellar_track is not None:
        age_Myr = age_yr / 1e6

        # Track type: 'spada' vs 'baraffe'
        track_type = mors_cfg.tracks

        if Teff is None:
            if track_type == 'spada':
                Teff = float(stellar_track.Value(age_Myr, 'Teff'))
            else:  # baraffe
                Teff = float(stellar_track.BaraffeStellarTeff(age_yr))
            log.info(
                f'PHOENIX: Assuming calculated effective temperature {Teff:.0f} K from {track_type} tracks'
            )

        if radius is None:
            if track_type == 'spada':
                radius = float(stellar_track.Value(age_Myr, 'Rstar'))  # [R_sun]
            else:  # baraffe
                radius = float(stellar_track.BaraffeStellarRadius(age_yr))  # [R_sun]
            log.info(
                f'PHOENIX: Assuming calculated stellar radius {radius:.2f} R_sun from {track_type} tracks'
            )

    # If log g is missing but we know mass and radius, compute it
    if logg is None and radius is not None:
        # Only use allowed mass range for the chosen tracks
        Mstar = float(star_cfg.mass)
        Mmin, Mmax = MASS_LIM[mors_cfg.tracks]

        if not (Mmin <= Mstar <= Mmax):
            msg = (
                f'Cannot compute log g: stellar mass {Mstar:.3f} Msun outside of '
                f'allowed range [{Mmin:.2f}, {Mmax:.2f}] for {mors_cfg.tracks} tracks. '
                'Please set log g manually, adjust the star mass or use a different stellar track.'
            )
            log.error(msg)

            UpdateStatusfile(handler.directories, 23)
            raise ValueError(msg)

        M_kg = Mstar * M_sun
        R_m = radius * R_sun

        g_cgs = const_G * M_kg / (R_m**2) * 100.0  # m/s^2 -> cm/s^2
        logg = float(np.log10(g_cgs))

        log.info(
            f'PHOENIX: Assuming calculated surface gravity log g = {logg:.2f} from mass and radius'
        )

    return {
        'Teff': Teff,
        'logg': logg,
        'radius': radius,
        'FeH': FeH,
        'alpha': alpha,
    }


def get_phoenix_modern_spectrum(
    handler: Proteus, stellar_track=None, age_yr: float | None = None
) -> Path:
    """
    Get a PHOENIX 'modern' spectrum scaled to 1 AU and return its path.

    Raw files in stellar_spectra/PHOENIX/FeH*_alpha*/
    Scaled 1 AU files in stellar_spectra/PHOENIX/1AU/
    """

    # parameters
    params = phoenix_params(handler, stellar_track=stellar_track, age_yr=age_yr)
    if params['Teff'] is None or params['radius'] is None or params['logg'] is None:
        raise ValueError('PHOENIX requires Teff, radius and log g to be known.')

    grid = phoenix_to_grid(
        FeH=params['FeH'], alpha=params['alpha'], Teff=params['Teff'], logg=params['logg']
    )

    Teff_g, logg_g, FeH_g, alpha_g = grid['Teff'], grid['logg'], grid['FeH'], grid['alpha']

    # Since Teff might be calculated from the stellar track, it might happen that the phoenix grid downloaded earlier (which was still agnostic of Teff) does not contain the necessary Teff, because of a forbidden Teff/alpha combination.
    # In that case, the appropriate grid (with alpha=0) will be downloaded. Here the user is informed about this.
    if Teff_g < 3500 or Teff_g > 8000:
        if abs(params['alpha']) > 1e-6:
            log.info('A new grid might be downloaded.')

    log.info(
        'PHOENIX: Using grid params Teff=%.0f K, logg=%.2f, [Fe/H]=%+0.1f, [alpha/M]=%+0.1f',
        Teff_g,
        logg_g,
        FeH_g,
        alpha_g,
    )
    log.info('')

    base_dir = GetFWLData() / 'stellar_spectra' / 'PHOENIX'

    feh_str = phoenix_param(FeH_g, kind='FeH')
    alpha_str = phoenix_param(alpha_g, kind='alpha')
    comp_dir = f'FeH{feh_str}_alpha{alpha_str}'

    raw_dir = base_dir / comp_dir
    au_dir = base_dir / '1AU'
    raw_dir.mkdir(parents=True, exist_ok=True)
    au_dir.mkdir(parents=True, exist_ok=True)

    raw_name = phoenix_filename(Teff_g, logg_g, FeH_g, alpha_g)
    raw_path = raw_dir / raw_name
    au_path = au_dir / raw_name

    # make sure raw PHOENIX file exists
    if not raw_path.exists():
        if not handler.config.params.offline:
            log.info(
                'PHOENIX file %s not found, downloading grid [Fe/H]=%+0.1f, [alpha/M]=%+0.1f',
                raw_name,
                FeH_g,
                alpha_g,
            )
            if not download_phoenix(alpha=alpha_g, FeH=FeH_g):
                raise RuntimeError(
                    f'Failed to download PHOENIX grid for [Fe/H]={FeH_g:+0.1f}, [alpha/M]={alpha_g:+0.1f}'
                )
            if not raw_path.exists():
                raise RuntimeError(f'PHOENIX file still missing after download: {raw_path}')
        else:
            log.error('Running in offline mode, but appropriate phoenix file is not available.')
            raise FileNotFoundError(f'PHOENIX file not found: {raw_path} (offline mode)')

    # scale from stellar surface to 1 AU and save to PHOENIX/1AU
    if (not au_path.exists()) or (au_path.stat().st_mtime < raw_path.stat().st_mtime):
        data = np.loadtxt(raw_path, comments='#')
        wl = data[:, 0]
        fl_surface = data[:, 1]

        R_m = params['radius'] * R_sun
        scale = (R_m / AU) ** 2
        fl_1au = fl_surface * scale

        header = (
            '# PHOENIX spectrum scaled to 1 AU\n'
            f'# Teff={Teff_g:.0f} logg={logg_g:.2f} [Fe/H]={FeH_g:+0.1f} [alpha/M]={alpha_g:+0.1f}\n'
            '# WL(nm)\tFlux(erg/cm^2/s/nm) at 1 AU'
        )

        np.savetxt(
            au_path,
            np.column_stack([wl, fl_1au]),
            header=header,
            comments='',
            fmt='%.8e',
            delimiter='\t',
        )
        log.info('Scaled PHOENIX spectrum to 1 AU.')

    return au_path
