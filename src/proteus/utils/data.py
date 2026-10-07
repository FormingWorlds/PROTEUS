from __future__ import annotations

import logging
import os
import subprocess as sp
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import interp1d

if TYPE_CHECKING:
    from proteus.config import Config

from proteus.utils.constants import PALEOS_REGISTRY_KEYS, VOLATILE_EOS_MAP
from proteus.utils.helper import (
    MissingDataError,
    energetics_eos_key,
    eos_components,
    paleos_companion_keys,
    resolve_fwl_data_dir,
    safe_rm,
    twophase_registry_key,
)
from proteus.utils.phoenix_helper import phoenix_param

log = logging.getLogger('fwl.' + __name__)

FWL_DATA_DIR = resolve_fwl_data_dir()

# Appended to the errors of a data file that an offline run cannot find.
RELOCATE_HINT = (
    'Data kept in the older FWL_DATA layout can be moved into place with `fwl-io relocate`.'
)

log.debug(f'FWL data location: {FWL_DATA_DIR}')


# Spectral file folders served by `proteus get spectral`. One entry per
# line, grouped by k-table set, ordered by band count. Every entry must
# have a matching manifest table (see proteus.data.spectral_file_key).
SPECTRAL_FILE_FOLDERS: tuple[str, ...] = (
    'Dayspring/16',
    'Dayspring/48',
    'Dayspring/256',
    'Dayspring/4096',
    'Frostflow/16',
    'Frostflow/48',
    'Frostflow/256',
    'Frostflow/4096',
    'Honeyside/16',
    'Honeyside/48',
    'Honeyside/256',
    'Honeyside/4096',
    'Oak/318',
)


def GetFWLData() -> Path:
    """
    Get path to FWL data directory on the disk
    """
    return Path(FWL_DATA_DIR).absolute()


def download_surface_albedos():
    """
    Download the surface reflectance data through fwl-io.

    The record pin and the file checksums come from the manifest PROTEUS ships,
    so the tables land in their version directory and are verified against the
    committed registry. AGNI needs these files, so a failed fetch raises.
    """
    from proteus.data import SURFACE_ALBEDOS_HAMMOND_2024, fetch_dataset

    fetch_dataset(SURFACE_ALBEDOS_HAMMOND_2024)


def download_scattering():
    """
    Download the monochromatic aerosol scattering data (``.mon``) through fwl-io.

    The tables land in the version directory of ``SCATTERING`` and are verified
    against the committed registry; DataverseNL serves them when Zenodo fails.
    """
    from proteus.data import SCATTERING, fetch_dataset

    fetch_dataset(SCATTERING)


def download_spectral_file(name: str, bands: str):
    """
    Download spectral file.

    Inputs :
        - name : str
            folder name (e.g. "Dayspring")
        - bands : str
            number of bands (e.g. "256")
    """
    # Check name and bands
    if not isinstance(name, str) or (len(name) < 1):
        raise ValueError('Must provide name of spectral file')
    if not isinstance(bands, str) or (len(bands) < 1):
        raise ValueError('Must provide number of bands in spectral file')

    from proteus.data import fetch_dataset, spectral_file_key

    folder = f'{name}/{bands}'
    if folder not in SPECTRAL_FILE_FOLDERS:
        raise ValueError(f'No data source mapping found for folder: {folder}')

    fetch_dataset(spectral_file_key(name, bands))


def download_spectral_files(name: str | None = None, bands: str | None = None):
    """
    Download spectral files, defaulting to the full set.

    Inputs :
        - name : str | None
            spectral file group (e.g. "Dayspring"). None selects every group.
        - bands : str | None
            number of bands (e.g. "256"). None selects every band count
            available for the group.
    """
    if name is None and bands is not None:
        raise ValueError(
            'Cannot select spectral files by band count alone; provide a group name'
        )
    if name is None:
        folders = SPECTRAL_FILE_FOLDERS
    elif bands is None:
        folders = tuple(f for f in SPECTRAL_FILE_FOLDERS if f.split('/')[0] == name)
        if not folders:
            known = sorted({f.split('/')[0] for f in SPECTRAL_FILE_FOLDERS})
            raise ValueError(f'Unknown spectral file group: {name}. Known groups: {known}')
    else:
        key = f'{name}/{bands}'
        if key not in SPECTRAL_FILE_FOLDERS:
            known_bands = sorted(
                f.split('/')[1] for f in SPECTRAL_FILE_FOLDERS if f.split('/')[0] == name
            )
            if not known_bands:
                known = sorted({f.split('/')[0] for f in SPECTRAL_FILE_FOLDERS})
                raise ValueError(f'Unknown spectral file group: {name}. Known groups: {known}')
            raise ValueError(
                f'Unknown band count {bands} for group {name}. Available: {known_bands}'
            )
        folders = (key,)

    for folder in folders:
        group, nbands = folder.split('/')
        download_spectral_file(group, nbands)


def download_phoenix(*, alpha: float = 0.0, FeH: float = 0.0, force: bool = False) -> bool:
    """Fetch one PHOENIX grid through fwl-io and unpack it.

    The dataset holds one zip archive per ([Fe/H], [alpha/M]) pair. Only the
    archive for the requested pair is fetched and verified against the registry
    checksum; the other archives of the record are not downloaded. The archive
    is unpacked into ``FeH<FeH>_alpha<alpha>/`` inside the dataset directory
    and kept beside it.

    Parameters
    ----------
    alpha : float
        Alpha-element enhancement [alpha/M] of the grid.
    FeH : float
        Metallicity [Fe/H] of the grid.
    force : bool
        Remove the unpacked grid and its archive first, so both are fetched again.

    Returns
    -------
    bool
        True if the unpacked grid is on disk.
    """
    from proteus.data import STELLAR_SPECTRA_PHOENIX, dataset_dir, fetch_dataset_file

    feh_str = phoenix_param(FeH, kind='FeH')
    alpha_str = phoenix_param(alpha, kind='alpha')
    zip_name = f'FeH{feh_str}_alpha{alpha_str}_phoenixMedRes_R05000.zip'
    lte_glob = 'LTE_T*_phoenixMedRes_R05000.txt'

    fwl_data = GetFWLData()
    base_dir = dataset_dir(STELLAR_SPECTRA_PHOENIX, data_root=fwl_data)
    grid_dir = base_dir / f'FeH{feh_str}_alpha{alpha_str}'

    if force:
        safe_rm(str(grid_dir))
        safe_rm(str(base_dir / zip_name))
    elif any(grid_dir.glob(lte_glob)):
        return True

    log.info(f'Fetching PHOENIX grid [Fe/H]={FeH:+0.1f}, [alpha/M]={alpha:+0.1f}: {zip_name}')
    try:
        zip_path = fetch_dataset_file(STELLAR_SPECTRA_PHOENIX, zip_name, data_root=fwl_data)
    except KeyError:
        log.error(f'No PHOENIX grid archive named {zip_name} in the dataset registry')
        return False
    except Exception as exc:  # noqa: BLE001 - the caller reports the failed download
        log.error(f'Failed to fetch PHOENIX grid {zip_name}: {exc}')
        return False

    # Unpack next to the final directory and rename, so an interrupted
    # extraction is never mistaken for a complete grid.
    partial_dir = grid_dir.with_name(grid_dir.name + '.partial')
    safe_rm(str(partial_dir))
    partial_dir.mkdir(parents=True, exist_ok=True)

    log.info(f'Unpacking PHOENIX zip: {zip_path.name} -> {grid_dir}')
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(partial_dir)
    except zipfile.BadZipFile as exc:
        log.error(f'PHOENIX archive {zip_path.name} cannot be unpacked: {exc}')
        safe_rm(str(partial_dir))
        return False

    if not any(partial_dir.glob(lte_glob)):
        log.error(f'Extraction completed but LTE files not found where expected: {partial_dir}')
        safe_rm(str(partial_dir))
        return False

    safe_rm(str(grid_dir))
    partial_dir.rename(grid_dir)
    return True


def download_muscles(stars: str | list[str] | None = None, *, force: bool = False) -> bool:
    """
    Download MUSCLES stellar spectrum(s) through fwl-io.

    Available to users who want to download specific MUSCLES spectra manually through the CLI.

    Parameters
    ----------
    stars:
        - None: download the whole MUSCLES catalogue
        - str: download one star (e.g. "trappist-1")
        - list[str]: download multiple stars
    force:
        Remove the requested files first so they are fetched again.

    Returns
    -------
    bool
        True if requested downloads succeeded (all of them, when list provided).
    """
    from proteus.data import (
        STELLAR_SPECTRA_MUSCLES,
        dataset_dir,
        fetch_dataset,
        fetch_dataset_file,
    )

    try:
        if stars is None:
            if force:
                safe_rm(dataset_dir(STELLAR_SPECTRA_MUSCLES))
            fetch_dataset(STELLAR_SPECTRA_MUSCLES)
            return True

        stars_list = [stars] if isinstance(stars, str) else list(stars)
        ok_all = True
        for star in stars_list:
            fname = f'{star}.txt'
            if force:
                safe_rm(dataset_dir(STELLAR_SPECTRA_MUSCLES) / fname)
            try:
                fetch_dataset_file(STELLAR_SPECTRA_MUSCLES, fname)
            except KeyError:
                log.error('No MUSCLES spectrum named %s in the dataset registry', star)
                ok_all = False
        return ok_all
    except Exception as exc:  # noqa: BLE001 -- the caller reports a failed download
        log.error('Failed to download MUSCLES stellar spectra: %s', exc)
        return False


def _melting_curve_keys() -> dict[str, str]:
    """Return the manifest key of each melting-curve name served by fwl-io.

    Returns
    -------
    dict
        Maps a ``interior_struct.melting_dir`` value to a dotted manifest key.
    """
    from proteus.data import (
        MELTING_MONTEUX_MINUS600,
        MELTING_MONTEUX_PLUS600,
        MELTING_WOLF_BOWER_2018,
    )

    return {
        'Monteux+600': MELTING_MONTEUX_PLUS600,
        'Monteux-600': MELTING_MONTEUX_MINUS600,
        'Wolf_Bower+2018': MELTING_WOLF_BOWER_2018,
    }


def resolve_melting_curve_files(
    melting_dir: str, data_root: Path | str | None = None
) -> tuple[Path, Path]:
    """Locate the P-T solidus and liquidus files of one melting curve.

    A name served through the manifest (``Monteux+600``, ``Monteux-600``,
    ``Wolf_Bower+2018``) resolves to ``solidus.dat`` and ``liquidus.dat`` in its
    checksum-verified dataset directory; a local directory
    ``interior_lookup_tables/Melting_curves/<melting_dir>`` with
    ``solidus_P-T.dat`` and ``liquidus_P-T.dat`` is used only when that dataset
    is not on disk. Any other name resolves to the local directory, which may
    hold ``solidus_P-T.dat`` and ``liquidus_P-T.dat`` or ``solidus.dat`` and
    ``liquidus.dat``.

    Parameters
    ----------
    melting_dir : str
        Value of ``interior_struct.melting_dir``.
    data_root : Path or str, optional
        Data root; defaults to ``FWL_DATA``.

    Returns
    -------
    tuple of Path
        Solidus and liquidus file paths; when no pair is on disk, the paths the
        caller reports as missing.
    """
    from proteus.data import dataset_dir

    root = GetFWLData() if data_root is None else Path(data_root)
    local = root / 'interior_lookup_tables' / 'Melting_curves' / melting_dir
    candidates = [(local / 'solidus_P-T.dat', local / 'liquidus_P-T.dat')]
    key = _melting_curve_keys().get(melting_dir)
    if key is None:
        candidates.append((local / 'solidus.dat', local / 'liquidus.dat'))
    else:
        folder = dataset_dir(key, data_root=root)
        candidates.insert(0, (folder / 'solidus.dat', folder / 'liquidus.dat'))
    for solidus, liquidus in candidates:
        if solidus.is_file() and liquidus.is_file():
            return solidus, liquidus
    return candidates[0]


def resolve_lookup_table_dir(data_root: Path | str | None = None) -> Path:
    """Return the directory of the Wolf and Bower 2018 P-S lookup tables.

    Parameters
    ----------
    data_root : Path or str, optional
        Data root; defaults to ``FWL_DATA``.

    Returns
    -------
    Path
        Versioned dataset directory of the tables. The directory is not
        checked to hold the files.
    """
    from proteus.data import LOOKUP_WOLF_BOWER_2018_1TPA, dataset_dir

    root = GetFWLData() if data_root is None else Path(data_root)
    return dataset_dir(LOOKUP_WOLF_BOWER_2018_1TPA, data_root=root)


def find_lookup_table_dir(data_root: Path | str | None = None) -> Path | None:
    """Return the fetched Wolf and Bower 2018 P-S table directory if populated.

    Parameters
    ----------
    data_root : Path or str, optional
        Data root; defaults to ``FWL_DATA``.

    Returns
    -------
    Path or None
        The dataset directory when it holds every SPIDER phase file and both
        P-S melting curves, else None (also when the data root cannot be
        created), so an interrupted fetch never shadows a complete table set.
    """
    from proteus.interior_energetics.common import (
        _SPIDER_EOS_MELTING_CURVES,
        _SPIDER_EOS_PHASE_FILES,
    )

    try:
        folder = resolve_lookup_table_dir(data_root)
    except OSError:
        return None
    names = _SPIDER_EOS_PHASE_FILES + _SPIDER_EOS_MELTING_CURVES
    if all((folder / name).is_file() for name in names):
        return folder
    return None


def download_interior_lookuptables(clean=False):
    """Fetch the Wolf and Bower 2018 melting curves.

    They are fetched through fwl-io into
    ``FWL_DATA/interior/melting_curves/wolf_bower_2018/r<record-id>/`` and read
    when ``interior_struct.melting_dir`` is ``Wolf_Bower+2018``.

    Parameters
    ----------
    clean : bool
        Remove the fetched dataset first, so it is fetched again.
    """
    from proteus.data import MELTING_WOLF_BOWER_2018, dataset_dir, fetch_dataset

    log.debug('Fetch basic interior lookup tables')
    root = GetFWLData()
    if clean:
        safe_rm(str(dataset_dir(MELTING_WOLF_BOWER_2018, data_root=root)))
    fetch_dataset(MELTING_WOLF_BOWER_2018, data_root=root)


def download_melting_curves(config: Config, clean: bool = False):
    """Ensure the melting curve named in the configuration is available.

    ``Monteux+600``, ``Monteux-600`` and ``Wolf_Bower+2018`` are fetched through
    fwl-io into ``FWL_DATA/interior/melting_curves/<key>/r<record-id>/``. Any
    other name must be a local directory
    ``interior_lookup_tables/Melting_curves/<melting_dir>`` that holds
    ``solidus_P-T.dat`` and ``liquidus_P-T.dat`` or ``solidus.dat`` and
    ``liquidus.dat``; nothing is fetched for it.

    Parameters
    ----------
    config : Config
        Configuration object; ``interior_struct.melting_dir`` names the curve.
    clean : bool
        For a served name, remove the fetched dataset and the local directory of
        that name first, so the curve is fetched again. A local directory of a
        name that is not served is never removed.

    Raises
    ------
    ValueError
        If the name is neither served through the manifest nor present as a
        local directory.
    """
    from proteus.data import dataset_dir, fetch_dataset

    log.debug('Fetch melting curve data')
    melting_dir = config.interior_struct.melting_dir
    if melting_dir is None:
        log.debug('melting_dir is None, skipping melting curve download')
        return

    root = GetFWLData()
    key = _melting_curve_keys().get(melting_dir)
    local = root / 'interior_lookup_tables' / 'Melting_curves' / melting_dir
    if key is not None:
        if clean:
            safe_rm(str(dataset_dir(key, data_root=root)))
            safe_rm(str(local))
        fetch_dataset(key, data_root=root)
        return

    solidus, liquidus = resolve_melting_curve_files(melting_dir, data_root=root)
    if not (solidus.is_file() and liquidus.is_file()):
        raise ValueError(
            f"No dataset serves melting_dir='{melting_dir}' and no local melting "
            f'curve files were found in: {local}'
        )
    log.debug('Melting curve data already present locally: %s', local)


def download_stellar_spectra(*, folders: tuple[str, ...] | None = None):
    """
    Download stellar spectra collections into ``FWL_DATA/star/spectra``.

    Notes
    -----
    Each collection is a manifest dataset and lands in its own version
    directory: ``star/spectra/solar/``, ``star/spectra/named/`` and
    ``star/spectra/muscles/``, each below an ``r<record-id>`` directory.

    Parameters
    ----------
    folders:
        Collections to download: ``'Named'``, ``'solar'`` or ``'MUSCLES'``.
        If None, downloads all three.

    Raises
    ------
    ValueError
        A requested collection is not known.
    """
    from proteus.data import (
        STELLAR_SPECTRA_MUSCLES,
        STELLAR_SPECTRA_NAMED,
        STELLAR_SPECTRA_SOLAR,
        fetch_dataset,
    )

    keys = {
        'Named': STELLAR_SPECTRA_NAMED,
        'solar': STELLAR_SPECTRA_SOLAR,
        'MUSCLES': STELLAR_SPECTRA_MUSCLES,
    }
    if folders is None:
        folders = tuple(keys)

    unknown = [folder for folder in folders if folder not in keys]
    if unknown:
        raise ValueError(
            f'Unknown stellar spectra collection(s) {unknown}; choose from {list(keys)}'
        )
    for folder in folders:
        fetch_dataset(keys[folder])


def _fetch_optional_dataset(key: str, desc: str) -> bool:
    """Fetch a dataset whose absence only costs a plot, reporting rather than raising.

    The overlays these datasets feed are decorative: a run must survive an
    unreachable mirror, an offline environment, or a data tree that cannot be
    resolved. fwl-io reports all of those by raising, and the reference data is
    fetched before the interior tables a run genuinely needs, so an escaping
    error would both abort the run and block the data that matters.

    Parameters
    ----------
    key : str
        Dotted manifest key of the dataset.
    desc : str
        Human-readable dataset name used in the log message.

    Returns
    -------
    bool
        Whether the dataset is now present.
    """
    from proteus.data import fetch_dataset

    try:
        fetch_dataset(key)
    except Exception as exc:  # noqa: BLE001 -- a decorative dataset never fails a run
        log.error('Failed to download %s: %s', desc, exc)
        log.error('Plots that use it will be skipped.')
        return False
    return True


def download_exoplanet_data():
    """
    Download the exoplanet catalogue through fwl-io.

    The record pin and the file checksums come from the manifest PROTEUS ships,
    so the catalogue lands in its version directory and is verified against the
    committed registry.
    """
    from proteus.data import EXOPLANET_REFERENCE

    return _fetch_optional_dataset(EXOPLANET_REFERENCE, 'exoplanet data')


def download_massradius_data():
    """
    Download the mass-radius relations through fwl-io.

    The record pin and the file checksums come from the fwl-io shared manifest,
    so the curves land in their version directory and are verified against its
    registry.
    """
    from proteus.data import MASS_RADIUS_ZENG_2019

    return _fetch_optional_dataset(MASS_RADIUS_ZENG_2019, 'mass radius data')


def download_stellar_tracks(track: str):
    """
    Download stellar evolution tracks through MORS.

    MORS fetches each track set through fwl-io from its Zenodo record, with its
    DataverseNL mirror as the fallback, and verifies the files against the
    committed registry.

    Parameters
    ----------
    track : str
        Track name ('Spada' or 'Baraffe')

    Raises
    ------
    ValueError
        ``track`` is not 'Spada' or 'Baraffe'; nothing is downloaded.
    FileNotFoundError
        MORS finished, but the track directory is missing or empty.
    OSError or fwl-io error
        A failed download, as MORS raises it.
    """
    from mors import data as mors_data

    accessors = {'Baraffe': mors_data.baraffe_data_dir, 'Spada': mors_data.spada_data_dir}
    if track not in accessors:
        raise ValueError(f'Unknown stellar track set {track!r}; choose from {list(accessors)}')
    log.debug(f'Downloading stellar evolution tracks: {track}')
    mors_data.DownloadEvolutionTracks(track)
    tracks_path = accessors[track]()
    if not (tracks_path.exists() and any(tracks_path.iterdir())):
        raise FileNotFoundError(f'Tracks directory empty or missing: {tracks_path}')
    log.info(f'Successfully downloaded {track} tracks via MORS')


def _fetch_errors() -> tuple[type[Exception], ...]:
    """Return what a fetch raises when a dataset cannot be obtained.

    Filesystem errors and fwl-io's own download errors; any other error (a
    stale fwl-io, a bug) is not a failed download and must propagate.

    Returns
    -------
    tuple of type
        ``(OSError, DownloadError, OfflineDataError, ArchiveError,
        MissingDataRootError)``.

    Raises
    ------
    RuntimeError
        The installed fwl-io lacks these classes, so it predates the floor.
    """
    try:
        from fwl_io import DownloadError, MissingDataRootError, OfflineDataError
        from fwl_io.archive import ArchiveError
    except ImportError as exc:
        from proteus.data import FWL_IO_FLOOR

        raise RuntimeError(
            f'the installed fwl-io is too old for PROTEUS; upgrade to fwl-io>={FWL_IO_FLOOR}.'
        ) from exc
    return (OSError, DownloadError, OfflineDataError, ArchiveError, MissingDataRootError)


def _attempt(desc: str, func, *args, **kwargs) -> bool:
    """Run one start-of-run fetch step, reporting a failure instead of raising.

    Each step is fetched independently, so an unreachable mirror for one
    dataset does not stop the others; a dataset that is still missing fails
    later, where it is read.

    Parameters
    ----------
    desc : str
        What the step fetches, for the log message.
    func : callable
        The fetch function, called with the remaining arguments.

    Returns
    -------
    bool
        Whether the step finished without a fetch error.
    """
    fetch_errors = _fetch_errors()
    try:
        func(*args, **kwargs)
    except fetch_errors as exc:
        log.warning('Problem when downloading/checking %s: %s', desc, exc)
        return False
    return True


def needs_spider_ps_tables(config) -> bool:
    """Return whether a run reads the Wolf and Bower P-S lookup set.

    SPIDER and Aragog read it unless Zalmoxis generates a PALEOS table set
    (:func:`proteus.utils.helper.generates_paleos_tables`). Under the dummy structure
    Aragog requires its phase-property lookup tables even with a PALEOS P-S set. A set
    interior_struct.eos_dir asks for it whenever SPIDER or Aragog run.

    Parameters
    ----------
    config : Config
        The run configuration.

    Returns
    -------
    bool
        True when the run needs the lookup set in FWL_DATA.
    """
    from proteus.utils.helper import generates_paleos_tables

    energetics = getattr(getattr(config, 'interior_energetics', None), 'module', None)
    if energetics not in ('spider', 'aragog'):
        return False
    struct = getattr(config, 'interior_struct', None)
    if getattr(struct, 'eos_dir', None) is not None:
        return True
    return not generates_paleos_tables(struct)


def _get_sufficient(config: Config, clean: bool = False):
    # Star stuff
    if config.star.module == 'mors':
        # Download only the spectra collections that this config may require.
        # (PHOENIX is handled separately via download_phoenix().)
        spec_src = config.star.mors.spectrum_source
        folders: list[str] = ['Named']
        if spec_src in (None, 'solar'):
            folders.append('solar')
        if spec_src in (None, 'muscles'):
            folders.append('MUSCLES')
        for folder in dict.fromkeys(folders):
            _attempt(f'stellar spectra {folder}', download_stellar_spectra, folders=(folder,))
        tracks = 'Spada' if config.star.mors.tracks == 'spada' else 'Baraffe'
        _attempt('stellar tracks', download_stellar_tracks, tracks)

    # Spectral files
    if config.atmos_clim.module in ('janus', 'agni'):
        # High-res file often used for post-processing
        _attempt('spectral file Honeyside/4096', download_spectral_file, 'Honeyside', '4096')

        # Skip the group/bands download when AGNI takes its spectral file
        # directly from the user (a custom path, or 'greygas').
        if config.atmos_clim.module == 'agni' and config.atmos_clim.agni.spectral_file:
            pass
        else:
            # Get the spectral file we need for this simluation
            from proteus.atmos_clim.common import get_spfile_name_and_bands

            group, bands = get_spfile_name_and_bands(config)
            if f'{group}/{bands}' in SPECTRAL_FILE_FOLDERS:
                _attempt(f'spectral file {group}/{bands}', download_spectral_file, group, bands)
            else:
                log.info(
                    'Spectral file %s/%s is in no manifest; it is read locally', group, bands
                )

    # Surface single-scattering data
    if config.atmos_clim.module == 'agni':
        _attempt('surface albedos', download_surface_albedos)

    # Aerosol scattering data
    if config.atmos_clim.module == 'agni' and config.atmos_clim.aerosols_enabled:
        _attempt('aerosol scattering data', download_scattering)

    # Exoplanet population data
    download_exoplanet_data()

    # Mass-radius reference data
    download_massradius_data()

    # Interior lookup tables (melting curves)
    if config.interior_energetics.module in ('aragog', 'spider'):
        # A configured curve that is still missing stops the run where it is read.
        _attempt('Wolf and Bower melting curves', download_interior_lookuptables, clean=clean)
        _attempt('melting curves', download_melting_curves, config, clean=clean)

    # P-S lookup set for SPIDER and Aragog, unless Zalmoxis generates PALEOS tables
    if needs_spider_ps_tables(config):
        _attempt('EOS lookup tables', download_eos_dynamic)

    # EOS for Zalmoxis (derived from struct.zalmoxis config, not struct.eos_dir)
    _attempt('Zalmoxis EOS tables', download_zalmoxis_eos_for_config, config)


def download_zalmoxis_eos_for_config(config) -> None:
    """Download the structure EOS tables that a config's Zalmoxis setup needs.

    Single extraction point for the per-layer EOS identifiers, shared by
    the start-of-run data check and ``proteus get interiordata``. The
    ``'none'`` ice-layer sentinel maps to ``''`` (no ice EOS). With
    ``dry_mantle = false`` it also fetches the tables of the dissolved
    volatiles that the structure solve adds to the mantle EOS. When the dummy
    structure builds a PALEOS P-S set it fetches the tables of every mantle
    component (the table check requires them) and the 2-phase pair, but no
    core tables; other dummy runs and structures need nothing.
    """
    struct_cfg = getattr(config, 'interior_struct', None)
    module = getattr(struct_cfg, 'module', None)
    anchor_pair = getattr(getattr(config, 'planet', None), 'temperature_mode', None) in (
        'liquidus_super',
        'adiabatic_from_cmb',
    )
    if module == 'dummy':
        energetics = getattr(getattr(config, 'interior_energetics', None), 'module', None)
        mantle = getattr(getattr(struct_cfg, 'zalmoxis', None), 'mantle_eos', None)
        if energetics in ('spider', 'aragog') and energetics_eos_key(mantle) in (
            PALEOS_REGISTRY_KEYS
        ):
            download_zalmoxis_eos(mantle_eos=mantle, with_core=False)
        return
    if module != 'zalmoxis':
        return
    zconf = struct_cfg.zalmoxis
    ice = getattr(zconf, 'ice_layer_eos', None)
    download_zalmoxis_eos(
        mantle_eos=getattr(zconf, 'mantle_eos', ''),
        core_eos=getattr(zconf, 'core_eos', ''),
        ice_layer_eos='' if ice in (None, 'none') else ice,
        volatile_eos=''
        if getattr(zconf, 'dry_mantle', True)
        else '+'.join(VOLATILE_EOS_MAP.values()),
        anchor_pair=anchor_pair,
    )


def download_sufficient_data(config: Config, clean: bool = False):
    """
    Download the required data based on the current options
    """

    log.info('Getting physical and reference data')

    if config.params.offline:
        # Don't try to get data
        log.warning('Running in offline mode. Will not check for reference data.')

    else:
        # Try to get data
        try:
            _get_sufficient(config, clean=clean)

        # Some issue. Usually due to lack of internet connection, but print the error
        #     anyway so that the user knows what happened.
        except OSError as e:
            log.warning('Problem when downloading/checking reference data')
            log.warning(str(e))

    log.info(' ')


def _none_dirs():
    from proteus.utils.helper import get_proteus_dir

    dirs = {'proteus': get_proteus_dir()}
    dirs['tools'] = os.path.join(dirs['proteus'], 'tools')
    return dirs


def get_socrates(dirs=None):
    """
    Download and install SOCRATES
    """

    log.info('Setting up SOCRATES')

    # None dirs
    if dirs is None:
        dirs = _none_dirs()

    # Get path. Lowercase matches install.sh, the CI action, and the
    # RAD_DIR convention; on case-sensitive filesystems an uppercase
    # path would create a second checkout next to the real one.
    workpath = os.path.join(dirs['proteus'], 'socrates')
    workpath = os.path.abspath(workpath)
    if os.path.isdir(workpath):
        log.debug('    already set up')
        return

    # Download, configure, and build
    log.debug('Running get_socrates.sh')
    cmd = [os.path.join(dirs['tools'], 'get_socrates.sh'), workpath]
    out = os.path.join(dirs['proteus'], 'nogit_setup_socrates.log')
    log.debug('    logging to %s' % out)
    with open(out, 'w') as hdl:
        sp.run(cmd, check=True, stdout=hdl, stderr=hdl)

    # Set environment
    os.environ['RAD_DIR'] = workpath
    log.debug('    done')


def get_petsc(dirs=None):
    """
    Download and install PETSc
    """

    log.info('Setting up PETSc')

    if dirs is None:
        dirs = _none_dirs()

    # Get path
    workpath = os.path.join(dirs['proteus'], 'petsc')
    workpath = os.path.abspath(workpath)
    if os.path.isdir(workpath):
        # already downloaded
        log.debug('    already set up')
        return

    # Download, configure, and build
    log.debug('Running get_petsc.sh')
    cmd = [os.path.join(dirs['tools'], 'get_petsc.sh'), workpath]
    out = os.path.join(dirs['proteus'], 'nogit_setup_petsc.log')
    log.debug('    logging to %s' % out)
    with open(out, 'w') as hdl:
        sp.run(cmd, check=True, stdout=hdl, stderr=hdl)

    log.debug('    done')


def get_spider(dirs=None):
    """
    Download and install SPIDER
    """

    if dirs is None:
        dirs = _none_dirs()

    # Need to install PETSc first
    get_petsc(dirs)

    log.info('Setting up SPIDER')

    # Get path
    workpath = os.path.join(dirs['proteus'], 'SPIDER')
    workpath = os.path.abspath(workpath)
    if os.path.isdir(workpath):
        # already downloaded
        log.debug('    already set up')
        return

    # Download, configure, and build
    log.debug('Running get_spider.sh')
    cmd = [os.path.join(dirs['tools'], 'get_spider.sh'), workpath]
    out = os.path.join(dirs['proteus'], 'nogit_setup_spider.log')
    log.debug('    logging to %s' % out)
    with open(out, 'w') as hdl:
        sp.run(cmd, check=True, stdout=hdl, stderr=hdl)

    log.debug('    done')


def download_eos_static():
    """Download static (Zalmoxis-only) EOS files.

    Fetches the Seager et al. (2007) EOS tables through fwl-io into
    ``FWL_DATA/interior/eos/seager_2007/r<record-id>/``.
    """
    download_Seager_EOS()


def download_eos_dynamic(eos_dir: str = 'WolfBower2018_MgSiO3'):
    """Fetch the Wolf and Bower 2018 P-S lookup tables through fwl-io.

    The tables land in ``FWL_DATA/interior/eos/dk09_1tpa_elec_free/
    mgsio3_wolf_bower_2018_1tpa/r19473625/``. The
    record provides the complete P-S set that both SPIDER and Aragog consume
    at runtime: 10 phase-property files (temperature, density, heat capacity,
    adiabatic gradient, thermal expansivity for melt and solid) plus the two
    P-S melting curves (``solidus_P-S.dat``, ``liquidus_P-S.dat``).

    After the fetch, the function warns if any of the 12 expected files is
    missing from the dataset directory.

    Parameters
    ----------
    eos_dir : str
        Name of the dynamic EOS folder. Reserved for future multi-EOS
        support; currently always resolves to the dataset above.
    """
    from proteus.data import LOOKUP_WOLF_BOWER_2018_1TPA, dataset_dir, fetch_dataset

    fwl_data = GetFWLData()
    fetch_dataset(LOOKUP_WOLF_BOWER_2018_1TPA, data_root=fwl_data)
    target_dir = dataset_dir(LOOKUP_WOLF_BOWER_2018_1TPA, data_root=fwl_data)

    from proteus.interior_energetics.common import (
        _SPIDER_EOS_MELTING_CURVES,
        _SPIDER_EOS_PHASE_FILES,
    )

    expected_files = _SPIDER_EOS_PHASE_FILES + _SPIDER_EOS_MELTING_CURVES
    missing = [f for f in expected_files if not (target_dir / f).is_file()]
    if missing:
        log.warning(
            'Lookup tables at %s are missing %d of %d expected files: %s. '
            'Aragog will fall back to the SPIDER submodule at runtime via '
            '_provide_spider_eos_tables.',
            target_dir,
            len(missing),
            len(expected_files),
            missing[:5],
        )
    else:
        log.debug(
            'Lookup tables complete: all %d files present at %s',
            len(expected_files),
            target_dir,
        )


def download_Seager_EOS():
    """Fetch the Seager EOS tables through fwl-io.

    The record pin and the file checksums come from the fwl-io shared manifest.
    Zalmoxis needs these tables for every Seager component, so a failed fetch
    raises.
    """
    from proteus.data import EOS_SEAGER_2007, fetch_dataset

    fetch_dataset(EOS_SEAGER_2007, data_root=FWL_DATA_DIR)


# Zalmoxis EOS download helpers: download_zalmoxis_eos fetches only the
# manifest files that the configured layer EOS read.

# Mantle EOS family prefixes whose registry entry carries the Seager iron
# table as its core fallback (see
# proteus.interior_struct.zalmoxis.load_zalmoxis_material_dictionaries).
# The start-of-run existence check requires every file those entries
# reference, so the data fetch must cover the fallback whenever such a
# mantle is selected. Flat single-table families (PALEOS:*, Chabrier:*)
# do not reference it. Kept in sync with the registry by a dedicated
# test in tests/utils/test_data.py.
SEAGER_FALLBACK_FAMILIES = (
    'WolfBower2018:',
    'RTPress100TPa:',
    'PALEOS-2phase:',
    'PALEOS-API-2phase:',
)


# Files Zalmoxis reads from each dataset whose record holds more than it needs.
_WOLF_BOWER_SOLID_FILES = ('density_solid.dat', 'heat_capacity_solid.dat')
_RTPRESS_EOS_FILES = (
    'density_melt.dat',
    'adiabat_temp_grad_melt.dat',
    'heat_capacity_melt.dat',
)
_WOLF_BOWER_EOS_FILES = (*_RTPRESS_EOS_FILES, *_WOLF_BOWER_SOLID_FILES)
_PALEOS_2PHASE_FILES = (
    'paleos_mgsio3_tables_pt_proteus_liquid.dat',
    'paleos_mgsio3_tables_pt_proteus_solid.dat',
)
_PALEOS_2PHASE_HIGHRES_FILES = (
    'paleos_mgsio3_tables_pt_proteus_liquid_highres.dat',
    'paleos_mgsio3_tables_pt_proteus_solid_highres.dat',
)


def download_zalmoxis_eos(
    mantle_eos: str,
    core_eos: str = '',
    ice_layer_eos: str = '',
    volatile_eos: str = '',
    anchor_pair: bool = False,
    with_core: bool = True,
):
    """Download Zalmoxis EOS data required for the given EOS configuration.

    Inspects the mantle, core, and ice layer EOS identifiers and downloads
    only the files needed. Every dataset is declared in the fwl-io shared
    manifest and lands under ``FWL_DATA/interior/eos/``.

    Parameters
    ----------
    mantle_eos : str
        Mantle EOS identifier (e.g. ``'PALEOS:MgSiO3'``, ``'WolfBower2018:MgSiO3'``).
    core_eos : str
        Core EOS identifier (e.g. ``'Seager2007:iron'``, ``'PALEOS:iron'``).
    ice_layer_eos : str
        Ice layer EOS identifier (e.g. ``'Seager2007:H2O'``, ``'PALEOS:H2O'``, or empty).
    volatile_eos : str
        ``+``-joined EOS components of dissolved volatiles, or empty.
    anchor_pair : bool
        Also fetch the MgSiO3 2-phase pair for any mantle, which the
        liquidus_super initial adiabat and the adiabatic_from_cmb entropy fallback read.
    with_core : bool
        Whether a structure solve reads a core EOS. False skips the Seager 2007
        iron core default and fallback; a Seager component is still fetched.
    """
    from proteus.data import (
        EOS_CHABRIER_2021,
        EOS_PALEOS_H2O,
        EOS_PALEOS_IRON,
        EOS_PALEOS_MGSIO3,
        EOS_PALEOS_MGSIO3_UNIFIED,
        EOS_RTPRESS_100TPA,
        EOS_WOLF_BOWER_2018,
        fetch_dataset,
        fetch_dataset_file,
    )

    failed = []

    def attempt(what, func, *args, **kwargs):
        # One unreachable dataset does not stop the others; the first failure is raised at the end.
        try:
            func(*args, **kwargs)
        except _fetch_errors() as exc:
            log.warning('Could not fetch %s: %s', what, exc)
            failed.append(exc)

    def fetch_files(key, names):
        for name in names:
            attempt(f'{name} from {key}', fetch_dataset_file, key, name, data_root=FWL_DATA_DIR)

    all_eos = [e for e in (mantle_eos, core_eos, ice_layer_eos, volatile_eos) if e]

    # Multi-component EOS strings: "PALEOS:MgSiO3:0.98+Chabrier:H:0.01"
    components = {c for eos_str in all_eos for c in eos_components(eos_str)}

    # Seager2007 static EOS: for a Seager component, and with a core solve for the
    # default Seager iron core (no core EOS) or a SEAGER_FALLBACK_FAMILIES mantle.
    if any(c.startswith('Seager2007') for c in components) or (
        with_core
        and (any(c.startswith(SEAGER_FALLBACK_FAMILIES) for c in components) or not core_eos)
    ):
        attempt('the Seager 2007 tables', download_eos_static)

    # A PALEOS mantle, and any mantle with a CMB or liquidus anchor, reads the 2-phase pair.
    components.update(paleos_companion_keys(mantle_eos))
    if anchor_pair:
        components.add(twophase_registry_key(mantle_eos))

    # WolfBower2018 T-dependent MgSiO3. The RTPress mantle pairs its melt tables
    # with the Wolf & Bower solid tables, so those files are needed for both.
    needs_wb = any(c.startswith('WolfBower2018') for c in components)
    needs_rtpress = any(c.startswith('RTPress100TPa') for c in components)
    if needs_wb:
        fetch_files(EOS_WOLF_BOWER_2018, _WOLF_BOWER_EOS_FILES)
    elif needs_rtpress:
        fetch_files(EOS_WOLF_BOWER_2018, _WOLF_BOWER_SOLID_FILES)

    # RTPress 100 TPa extended melt
    if needs_rtpress:
        fetch_files(EOS_RTPRESS_100TPA, _RTPRESS_EOS_FILES)

    # PALEOS unified tables: one dataset per material.
    unified = {
        'PALEOS:iron': (EOS_PALEOS_IRON, 'paleos_iron_eos_table_pt.dat'),
        'PALEOS:MgSiO3': (EOS_PALEOS_MGSIO3_UNIFIED, 'paleos_mgsio3_eos_table_pt.dat'),
        'PALEOS:H2O': (EOS_PALEOS_H2O, 'paleos_water_eos_table_pt.dat'),
    }
    for component, (key, table) in unified.items():
        if component in components:
            fetch_files(key, (table,))

    # PALEOS 2-phase MgSiO3 (separate solid/liquid), one record holding the
    # 150 pts/decade pair and the 600 pts/decade (-highres) pair; fetch only the
    # selected pair, since the highres pair alone is about 1.3 GB.
    if 'PALEOS-2phase:MgSiO3' in components:
        fetch_files(EOS_PALEOS_MGSIO3, _PALEOS_2PHASE_FILES)
    if 'PALEOS-2phase:MgSiO3-highres' in components:
        fetch_files(EOS_PALEOS_MGSIO3, _PALEOS_2PHASE_HIGHRES_FILES)

    # Chabrier H/He
    if any(c.startswith('Chabrier') for c in components):
        attempt(
            'the Chabrier 2021 tables', fetch_dataset, EOS_CHABRIER_2021, data_root=FWL_DATA_DIR
        )

    # Defensive warn for any component key that no handler above recognised.
    # PALEOS-API:* and PALEOS-API-2phase:* are intentionally not downloaded
    # (generated locally from upstream paleos at runtime) but are valid.
    known_prefixes = (
        'Seager2007:',
        'WolfBower2018:',
        'RTPress100TPa:',
        'Chabrier:',
        'PALEOS-API:',
        'PALEOS-API-2phase:',
    )
    known_exact = {
        'PALEOS-2phase:MgSiO3',
        'PALEOS-2phase:MgSiO3-highres',
        'PALEOS:iron',
        'PALEOS:MgSiO3',
        'PALEOS:H2O',
    }
    for c in components:
        if c in known_exact:
            continue
        if any(c.startswith(p) for p in known_prefixes):
            continue
        log.warning(
            'download_zalmoxis_eos: no handler for component %r '
            '(typo or unsupported EOS family?); no data downloaded for it',
            c,
        )
    if failed:
        raise failed[0]


def load_melting_curve(melt_file):
    """
    Loads melting curve data for MgSiO3 from a text file.
    Parameters:
        melt_file: Path to the melting curve data file
    Returns:
        interp_func: Interpolation function for T(P)
    """
    try:
        data = np.loadtxt(melt_file, comments='#')
        pressures = data[:, 0]  # in Pa
        temperatures = data[:, 1]  # in K
        interp_func = interp1d(
            pressures, temperatures, kind='linear', bounds_error=False, fill_value=np.nan
        )
        return interp_func
    except Exception as e:
        log.error('Error loading melting curve data: %s', e)
        return None


def get_zalmoxis_melting_curves(config: Config):
    """Load solidus and liquidus T(P) melting curves for Zalmoxis.

    Reads the melting curve files that :func:`resolve_melting_curve_files`
    locates for ``config.interior_struct.melting_dir``.

    Parameters
    ----------
    config : Config
        Configuration object containing interior settings.

    Returns
    -------
    tuple
        (solidus_func, liquidus_func) interpolation functions T(P) [K].
    """
    if config.interior_struct.melting_dir is None:
        return None
    solidus_file, liquidus_file = resolve_melting_curve_files(
        config.interior_struct.melting_dir, data_root=FWL_DATA_DIR
    )
    for melting_file in (solidus_file, liquidus_file):
        if not melting_file.is_file():
            raise MissingDataError(
                f'Melting curve file not found: {melting_file}. '
                f"Check interior_struct.melting_dir='{config.interior_struct.melting_dir}', or fetch "
                f'it with `proteus get interiordata --config-path <config.toml>`. {RELOCATE_HINT}'
            )
    solidus_func = load_melting_curve(solidus_file)
    liquidus_func = load_melting_curve(liquidus_file)
    return (solidus_func, liquidus_func)


def get_zalmoxis_EOS():
    """Build and return material properties dictionaries for Zalmoxis.

    The Seager2007 tables are fetched through fwl-io into their versioned
    dataset directory, as are the Wolf & Bower 2018 and RTPress 100 TPa tables.

    Returns
    -------
    tuple
        Four dictionaries: iron/silicate, iron/T-dep silicate (WolfBower2018),
        water planet, and iron/RTPress100TPa silicate EOS. T-dep dicts include
        ``cp_file`` entries for heat capacity tables when the files exist.
    """
    from proteus.data import (
        EOS_RTPRESS_100TPA,
        EOS_SEAGER_2007,
        EOS_WOLF_BOWER_2018,
        dataset_dir,
    )

    seager_folder = dataset_dir(EOS_SEAGER_2007, data_root=FWL_DATA_DIR)
    if not (seager_folder / 'eos_seager07_iron.txt').exists():
        log.debug('Get EOS material properties from Seager et al. (2007)')
        download_eos_static()

    wb_folder = dataset_dir(EOS_WOLF_BOWER_2018, data_root=FWL_DATA_DIR)

    # Iron/silicate (Seager 2007)
    material_properties_iron_silicate_planets = {
        'core': {'eos_file': seager_folder / 'eos_seager07_iron.txt'},
        'mantle': {'eos_file': seager_folder / 'eos_seager07_silicate.txt'},
    }

    # Iron core (Seager 2007) + T-dependent silicate mantle (Wolf & Bower 2018)
    wb_cp_melt = wb_folder / 'heat_capacity_melt.dat'
    wb_cp_solid = wb_folder / 'heat_capacity_solid.dat'
    wb_adiabat_grad = wb_folder / 'adiabat_temp_grad_melt.dat'
    material_properties_iron_Tdep_silicate_planets = {
        'core': {'eos_file': seager_folder / 'eos_seager07_iron.txt'},
        'melted_mantle': {
            'eos_file': wb_folder / 'density_melt.dat',
            **({'cp_file': wb_cp_melt} if wb_cp_melt.is_file() else {}),
            **({'adiabat_grad_file': wb_adiabat_grad} if wb_adiabat_grad.is_file() else {}),
        },
        'solid_mantle': {
            'eos_file': wb_folder / 'density_solid.dat',
            **({'cp_file': wb_cp_solid} if wb_cp_solid.is_file() else {}),
        },
    }

    # Water planets (Seager 2007)
    material_properties_water_planets = {
        'core': {'eos_file': seager_folder / 'eos_seager07_iron.txt'},
        'mantle': {'eos_file': seager_folder / 'eos_seager07_silicate.txt'},
        'ice_layer': {'eos_file': seager_folder / 'eos_seager07_water.txt'},
    }

    # RTPress100TPa melt EOS with WolfBower2018 solid
    rt_folder = dataset_dir(EOS_RTPRESS_100TPA, data_root=FWL_DATA_DIR)
    if not rt_folder.exists():
        log.warning(
            'RTPress100TPa EOS folder not found at %s. '
            'RTPress100TPa:MgSiO3 will fail if used. '
            'Run `proteus get` to download EOS data.',
            rt_folder,
        )
    rt_cp_melt = rt_folder / 'heat_capacity_melt.dat'
    if not rt_cp_melt.is_file():
        log.warning(
            'RTPress100TPa melt Cp table not found at %s. '
            'Adiabatic mode will fall back to constant Cp for this EOS.',
            rt_cp_melt,
        )
    rt_adiabat_grad = rt_folder / 'adiabat_temp_grad_melt.dat'
    material_properties_iron_RTPress100TPa_silicate_planets = {
        'core': {'eos_file': seager_folder / 'eos_seager07_iron.txt'},
        'melted_mantle': {
            'eos_file': rt_folder / 'density_melt.dat',
            **({'cp_file': rt_cp_melt} if rt_cp_melt.is_file() else {}),
            **({'adiabat_grad_file': rt_adiabat_grad} if rt_adiabat_grad.is_file() else {}),
        },
        'solid_mantle': {
            'eos_file': wb_folder / 'density_solid.dat',
            **({'cp_file': wb_cp_solid} if wb_cp_solid.is_file() else {}),
        },
    }

    # Convert Path objects to str for compatibility with Zalmoxis internals
    # (which use os.path.join and str-based cache keys throughout).
    def _paths_to_str(d):
        return {
            layer: {k: str(v) if isinstance(v, Path) else v for k, v in props.items()}
            for layer, props in d.items()
        }

    return (
        _paths_to_str(material_properties_iron_silicate_planets),
        _paths_to_str(material_properties_iron_Tdep_silicate_planets),
        _paths_to_str(material_properties_water_planets),
        _paths_to_str(material_properties_iron_RTPress100TPa_silicate_planets),
    )


def get_Seager_EOS():
    """Backward-compatible Seager EOS helper.

    Returns the Seager et al. (2007) EOS material property dictionaries for
    iron/silicate and water planets. This mirrors the original
    ``get_Seager_EOS`` API that older code and tests expect.

    The implementation reuses :func:`get_zalmoxis_EOS` and returns only the
    iron/silicate and water dictionaries.
    """

    iron_silicate, _iron_Tdep, water, _iron_RTPress = get_zalmoxis_EOS()
    return iron_silicate, water
