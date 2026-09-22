"""Datasets PROTEUS fetches through fwl-io.

The datasets PROTEUS owns are declared in ``proteus_manifest.toml`` beside this
module, one table per dataset, with a committed registry of file checksums next
to it. fwl-io derives each dataset's location from its manifest key and places
it in a version directory named for the pinned Zenodo record, so a dataset lands
in ``<FWL_DATA>/<key-as-path>/r<record-id>`` and the pin, the location, and the
checksums have a single source of truth.

Readers resolve a dataset directory through :func:`dataset_dir` rather than
joining a path by hand, so the version segment stays an implementation detail of
the pin. Datasets absent from the manifest are provisioned by the downloader in
:mod:`proteus.utils.data` instead.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

# Manifest keys of the datasets declared beside this module. Readers refer to
# these constants rather than repeating the dotted strings, so a key rename is a
# single edit here and in the manifest.
EXOPLANET_REFERENCE = 'observe.exoplanet_reference'
MASS_RADIUS_ZENG_2019 = 'observe.mass_radius.zeng_2019'
SURFACE_ALBEDOS_HAMMOND_2024 = 'atmos_clim.surface_albedos.hammond_2024'
EOS_SEAGER_2007 = 'interior_struct.eos.seager_2007'
EOS_WOLF_BOWER_2018 = 'interior_struct.eos.wolf_bower_2018'
EOS_RTPRESS_100TPA = 'interior_struct.eos.rtpress_100tpa'
EOS_PALEOS_MGSIO3_2PHASE = 'interior_struct.eos.paleos_mgsio3_2phase'
EOS_PALEOS_MGSIO3_2PHASE_HIGHRES = 'interior_struct.eos.paleos_mgsio3_2phase_highres'
EOS_PALEOS_UNIFIED = 'interior_struct.eos.paleos_unified'
EOS_CHABRIER_2021 = 'interior_struct.eos.chabrier_2021'
LOOKUP_WOLF_BOWER_2018_1TPA = 'interior_struct.lookup.wolf_bower_2018_1tpa'
MELTING_MONTEUX_PLUS600 = 'interior_struct.melting_curves.monteux_plus600'
MELTING_MONTEUX_MINUS600 = 'interior_struct.melting_curves.monteux_minus600'
MELTING_WOLF_BOWER_2018 = 'interior_struct.melting_curves.wolf_bower_2018'
STELLAR_SPECTRA_SOLAR = 'stellar_spectra.solar'
STELLAR_SPECTRA_NAMED = 'stellar_spectra.named'
STELLAR_SPECTRA_MUSCLES = 'stellar_spectra.muscles'
STELLAR_SPECTRA_PHOENIX = 'stellar_spectra.phoenix'


def spectral_file_key(group: str, bands: str | int) -> str:
    """Return the manifest key of one spectral-file dataset.

    Parameters
    ----------
    group : str
        Spectral file group, e.g. ``Dayspring``.
    bands : str or int
        Number of bands, e.g. ``256``.

    Returns
    -------
    str
        Dotted manifest key, e.g. ``atmos_clim.spectral_files.dayspring_256``.
        The key is not checked against the manifest; an unknown pair raises
        ``KeyError`` when the dataset is resolved.
    """
    return f'atmos_clim.spectral_files.{str(group).lower()}_{bands}'


# The oldest fwl-io that reads this manifest schema. An older fwl-io reads the
# manifest as malformed rather than as a version mismatch, so the load names
# which side is out of date. The fwl-io requirement in pyproject.toml must be at
# least this version; the test suite enforces the relation.
FWL_IO_FLOOR = '26.7.25'


def manifest_path() -> Path:
    """Return the path of the dataset manifest fwl-io reads.

    This is the target of the ``fwl_io.manifests`` entry point, so fwl-io
    discovers PROTEUS's datasets from an installed package without PROTEUS
    having to register them at import time.

    Returns
    -------
    Path
        Location of ``proteus_manifest.toml`` inside the installed package.
    """
    return Path(__file__).parent / 'proteus_manifest.toml'


def _fwl_io_derives_the_location() -> bool:
    """Report whether the installed fwl-io derives a dataset location from its key.

    The question is whether ``subdir`` is still a manifest field, so the answer
    is read off the dataset fields rather than off how the attribute happens to
    be implemented. Only positive evidence of the older schema counts: an fwl-io
    that cannot be introspected is reported as current, so a manifest error is
    never blamed on a version mismatch that has not been demonstrated.

    Returns
    -------
    bool
        True when the installed fwl-io derives locations from manifest keys.
    """
    try:
        from fwl_io.manifest import Dataset

        return 'subdir' not in {field.name for field in dataclasses.fields(Dataset)}
    except Exception:
        return True


def _data_root() -> Path:
    """Return the FWL_DATA root PROTEUS resolves its data against.

    Imported here rather than at module scope so that :mod:`proteus.utils.data`
    can delegate to this module without an import cycle, and so the root is read
    when a dataset is resolved rather than frozen at import.

    Returns
    -------
    Path
        Root of the local reference-data tree.
    """
    from proteus.utils.data import GetFWLData

    return GetFWLData()


def _dataset(key: str):
    """Return one dataset declared in the shipped manifest.

    Parameters
    ----------
    key : str
        Dotted manifest key of the dataset, e.g. ``observe.exoplanet_reference``.

    Returns
    -------
    fwl_io.manifest.Dataset
        The declared dataset, carrying its Zenodo pin and registry.

    Raises
    ------
    RuntimeError
        The installed fwl-io predates the manifest schema PROTEUS ships.
    KeyError
        The key is absent from the manifest.
    """
    from fwl_io import load_manifest

    try:
        datasets = {ds.key: ds for ds in load_manifest(manifest_path())}
    except ValueError as exc:
        if _fwl_io_derives_the_location():
            raise
        raise RuntimeError(
            f'fwl-io could not read the manifest PROTEUS ships ({exc}); the installed '
            f'fwl-io predates the manifest schema: upgrade to fwl-io>={FWL_IO_FLOOR}.'
        ) from exc
    return datasets[key]


def _fetcher(key: str, data_root: str | Path | None = None):
    """Build an fwl-io fetcher for one declared dataset.

    Parameters
    ----------
    key : str
        Dotted manifest key of the dataset.
    data_root : str or Path, optional
        Reference-data tree to resolve against. Defaults to the tree PROTEUS
        resolves from the environment.

    Returns
    -------
    fwl_io.fetch.Fetcher
        Fetcher bound to the pinned record and the committed registry.
    """
    from fwl_io import create_fetcher

    ds = _dataset(key)
    return create_fetcher(
        subdir=ds.subdir,
        zenodo=ds.zenodo,
        dataverse=ds.dataverse,
        registry=ds.registry(),
        data_root=_data_root() if data_root is None else Path(data_root),
        extract=ds.extract,
    )


def dataset_dir(key: str, data_root: str | Path | None = None) -> Path:
    """Return the version directory holding one dataset's files.

    Resolving the path creates the data root if it is absent (an fwl-io side
    effect) but downloads nothing; call :func:`fetch_dataset` to populate it.
    The ``r<record-id>`` segment comes from the manifest pin, so callers never
    join it themselves.

    Parameters
    ----------
    key : str
        Dotted manifest key of the dataset.
    data_root : str or Path, optional
        Reference-data tree to resolve against. Defaults to the tree PROTEUS
        resolves from the environment.

    Returns
    -------
    Path
        Directory the dataset's files are placed in.

    Raises
    ------
    RuntimeError
        fwl-io resolved a directory without a version segment, which would put
        the files one level above where every reader looks for them.
    """
    fetcher = _fetcher(key, data_root=data_root)
    if getattr(fetcher, 'version_dir', None) is None:
        raise RuntimeError(
            f'fwl-io resolved an unversioned directory {fetcher.target_dir} for dataset '
            f'{key!r}; the files are expected under an r<record-id> version directory.'
        )
    return fetcher.target_dir


def fetch_dataset(key: str, data_root: str | Path | None = None) -> list[Path]:
    """Fetch one declared dataset, verifying every file against the registry.

    The fetch is idempotent: files already present with a matching checksum are
    left alone, so calling this on a populated tree costs no download.

    Parameters
    ----------
    key : str
        Dotted manifest key of the dataset.
    data_root : str or Path, optional
        Reference-data tree to fetch into. Defaults to the tree PROTEUS resolves
        from the environment.

    Returns
    -------
    list of Path
        The verified files of the dataset.
    """
    return _fetcher(key, data_root=data_root).fetch_all()


def fetch_dataset_file(key: str, name: str, data_root: str | Path | None = None) -> Path:
    """Fetch one file of a declared dataset, verifying it against the registry.

    Datasets that hold many independent files (one observed spectrum per star)
    can be fetched file by file so a reader downloads only what it needs. The
    fetch is idempotent: a file already present with a matching checksum is left
    alone.

    Parameters
    ----------
    key : str
        Dotted manifest key of the dataset.
    name : str
        File name as listed in the dataset registry.
    data_root : str or Path, optional
        Reference-data tree to fetch into. Defaults to the tree PROTEUS resolves
        from the environment.

    Returns
    -------
    Path
        The verified file.

    Raises
    ------
    KeyError
        ``name`` is not in the dataset registry.
    """
    return _fetcher(key, data_root=data_root).fetch(name)
