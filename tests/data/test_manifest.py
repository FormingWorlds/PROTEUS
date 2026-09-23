"""Tests for :mod:`proteus.data`, the datasets PROTEUS fetches through fwl-io.

Contract clauses exercised here:

- the shipped manifest declares exactly the datasets PROTEUS owns, with version
  DOIs, and each dataset's location is derived from its manifest key;
- every declared dataset has a committed registry of file checksums beside the
  manifest, and the manifest plus registries travel in the wheel;
- a dataset resolves into its ``r<record-id>`` version directory, never the bare
  location one level above it, which is where every reader looks;
- an fwl-io too old to read the shipped manifest is named as the stale side,
  while a genuine manifest defect surfaces as itself.

See ``docs/How-to/testing.md`` and ``docs/Explanations/test_framework.md``.
"""

from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path

import pytest

from proteus.data import (
    EOS_CHABRIER_2021,
    EOS_PALEOS_H2O,
    EOS_PALEOS_IRON,
    EOS_PALEOS_MGSIO3,
    EOS_PALEOS_MGSIO3_UNIFIED,
    EOS_RTPRESS_100TPA,
    EOS_SEAGER_2007,
    EOS_WOLF_BOWER_2018,
    EXOPLANET_REFERENCE,
    FWL_IO_FLOOR,
    LOOKUP_WOLF_BOWER_2018_1TPA,
    MASS_RADIUS_ZENG_2019,
    MELTING_MONTEUX_MINUS600,
    MELTING_MONTEUX_PLUS600,
    MELTING_WOLF_BOWER_2018,
    STELLAR_SPECTRA_MUSCLES,
    STELLAR_SPECTRA_NAMED,
    STELLAR_SPECTRA_PHOENIX,
    STELLAR_SPECTRA_SOLAR,
    SURFACE_ALBEDOS_HAMMOND_2024,
    _dataset,
    _fwl_io_derives_the_location,
    dataset_dir,
    fetch_dataset,
    manifest_path,
    spectral_file_key,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# The pinned records, repeated here so a silent re-pin of the manifest fails a
# test rather than quietly moving every reader to a different deposit.
EXOPLANET_RECORD = '15727878'
ZENG_2019_RECORD = '15727899'
HAMMOND_2024_RECORD = '15880455'
SEAGER_2007_RECORD = '15727998'
SOLAR_RECORD = '17981836'
NAMED_RECORD = '15721440'
MUSCLES_RECORD = '17802209'
PHOENIX_RECORD = '17674612'
WOLF_BOWER_RECORD = '17417017'
RTPRESS_RECORD = '18819027'
PALEOS_2PHASE_RECORD = '19680050'
PALEOS_UNIFIED_RECORD = '22776069'
CHABRIER_RECORD = '19135021'
LOOKUP_WOLF_BOWER_2018_1TPA_RECORD = '19473625'
MELTING_MONTEUX_PLUS600_RECORD = '15728091'
MELTING_MONTEUX_MINUS600_RECORD = '15728138'
MELTING_WOLF_BOWER_2018_RECORD = '15728072'

# Datasets PROTEUS reads from the fwl-io shared manifest: key -> (subdir, record).
SHARED_DATASETS = {
    EOS_WOLF_BOWER_2018: ('interior/eos/wolf_bower_2018_1tpa', WOLF_BOWER_RECORD),
    EOS_RTPRESS_100TPA: ('interior/eos/rtpress_melt_100tpa', RTPRESS_RECORD),
    EOS_PALEOS_MGSIO3: ('interior/eos/paleos_mgsio3', PALEOS_2PHASE_RECORD),
    EOS_PALEOS_IRON: ('interior/eos/paleos_iron', PALEOS_UNIFIED_RECORD),
    EOS_PALEOS_MGSIO3_UNIFIED: ('interior/eos/paleos_mgsio3_unified', PALEOS_UNIFIED_RECORD),
    EOS_PALEOS_H2O: ('interior/eos/paleos_h2o', PALEOS_UNIFIED_RECORD),
    EOS_CHABRIER_2021: ('interior/eos/chabrier_2021_hhe', CHABRIER_RECORD),
    LOOKUP_WOLF_BOWER_2018_1TPA: (
        'interior/eos/dk09_1tpa_elec_free/mgsio3_wolf_bower_2018_1tpa',
        LOOKUP_WOLF_BOWER_2018_1TPA_RECORD,
    ),
    MELTING_MONTEUX_PLUS600: (
        'interior/melting_curves/monteux_plus_600',
        MELTING_MONTEUX_PLUS600_RECORD,
    ),
    MELTING_MONTEUX_MINUS600: (
        'interior/melting_curves/monteux_minus_600',
        MELTING_MONTEUX_MINUS600_RECORD,
    ),
    MELTING_WOLF_BOWER_2018: (
        'interior/melting_curves/wolf_bower_2018',
        MELTING_WOLF_BOWER_2018_RECORD,
    ),
    STELLAR_SPECTRA_SOLAR: ('star/spectra/solar', SOLAR_RECORD),
    STELLAR_SPECTRA_NAMED: ('star/spectra/named', NAMED_RECORD),
    STELLAR_SPECTRA_MUSCLES: ('star/spectra/muscles', MUSCLES_RECORD),
    STELLAR_SPECTRA_PHOENIX: ('star/spectra/phoenix', PHOENIX_RECORD),
}

REPO_ROOT = Path(__file__).resolve().parents[2]

# Spectral-file datasets: (group, bands) -> Zenodo record, one dataset each.
SPECTRAL_RECORDS = {
    ('Frostflow', '16'): '15799743',
    ('Frostflow', '48'): '15696415',
    ('Frostflow', '256'): '15799754',
    ('Frostflow', '4096'): '15799776',
    ('Dayspring', '16'): '15799318',
    ('Dayspring', '48'): '15721749',
    ('Dayspring', '256'): '15799474',
    ('Dayspring', '4096'): '15799495',
    ('Honeyside', '16'): '15799607',
    ('Honeyside', '48'): '15799652',
    ('Honeyside', '256'): '15799731',
    ('Honeyside', '4096'): '15696457',
    ('Oak', '318'): '15743843',
}

# Datasets declared in proteus_manifest.toml: only data PROTEUS alone reads.
_OWNED_KEYS = {
    EXOPLANET_REFERENCE,
    MASS_RADIUS_ZENG_2019,
    SURFACE_ALBEDOS_HAMMOND_2024,
    EOS_SEAGER_2007,
}


def _pyproject() -> dict:
    """Return the parsed pyproject.toml of the repository under test."""
    with open(REPO_ROOT / 'pyproject.toml', 'rb') as handle:
        return tomllib.load(handle)


def _shared_manifest() -> dict:
    """Return the fwl-io shared manifest, keyed by dataset key."""
    from fwl_io import load_manifest
    from fwl_io.manifest import shared_manifest_path

    return {ds.key: ds for ds in load_manifest(shared_manifest_path())}


def test_manifest_declares_the_datasets():
    """The PROTEUS manifest declares exactly the datasets only PROTEUS reads.

    Both the key set and each dataset's derived location are pinned: fwl-io
    turns the dotted key into the on-disk path, so a key edit silently relocates
    the data unless the mapping is asserted.
    """
    from fwl_io import load_manifest

    datasets = {ds.key: ds for ds in load_manifest(manifest_path())}

    assert set(datasets) == _OWNED_KEYS
    assert datasets[EXOPLANET_REFERENCE].subdir == 'observe/exoplanet_reference'
    assert datasets[MASS_RADIUS_ZENG_2019].subdir == 'observe/mass_radius/zeng_2019'
    assert (
        datasets[SURFACE_ALBEDOS_HAMMOND_2024].subdir
        == 'atmos_clim/surface_albedos/hammond_2024'
    )
    assert datasets[EOS_SEAGER_2007].subdir == 'interior_struct/eos/seager_2007'
    assert datasets[EOS_SEAGER_2007].zenodo == f'10.5281/zenodo.{SEAGER_2007_RECORD}'
    assert datasets[SURFACE_ALBEDOS_HAMMOND_2024].zenodo == (
        f'10.5281/zenodo.{HAMMOND_2024_RECORD}'
    )
    assert datasets[EXOPLANET_REFERENCE].zenodo == f'10.5281/zenodo.{EXOPLANET_RECORD}'
    assert datasets[MASS_RADIUS_ZENG_2019].zenodo == f'10.5281/zenodo.{ZENG_2019_RECORD}'
    # All are PROTEUS-owned, so "proteus" has to appear in required_by or
    # "fwl-io fetch proteus" would skip them.
    for ds in datasets.values():
        assert 'proteus' in [model.lower() for model in ds.required_by]


def test_shared_datasets_resolve_through_the_fwl_io_manifest():
    """Every multi-model dataset PROTEUS reads comes from the fwl-io shared manifest.

    The subdir and record are pinned per key, so a key that drifts from the
    shared manifest, or a re-pin there, fails here rather than at a user's fetch.
    """
    shared = _shared_manifest()

    for key, (subdir, record) in SHARED_DATASETS.items():
        assert key in shared, f'{key} is not declared in the fwl-io shared manifest'
        assert _dataset(key).subdir == subdir
        assert _dataset(key).zenodo == f'10.5281/zenodo.{record}'
    for (group, bands), record in SPECTRAL_RECORDS.items():
        assert shared[spectral_file_key(group, bands)].zenodo == f'10.5281/zenodo.{record}'
    # Discrimination: none of these keys is PROTEUS-owned, so a lookup that only
    # read proteus_manifest.toml would have raised above.
    assert not set(SHARED_DATASETS) & _OWNED_KEYS


def test_no_record_is_declared_in_both_manifests():
    """A Zenodo record is pinned in one manifest only.

    Declaring a shared record again in proteus_manifest.toml under another key
    would place a second copy elsewhere in FWL_DATA, and ``fwl-io relocate``
    would move a legacy tree to the shared key that PROTEUS then does not read.
    """
    from fwl_io import load_manifest

    owned_datasets = load_manifest(manifest_path())
    owned = {ds.zenodo for ds in owned_datasets}
    shared = {ds.zenodo for ds in _shared_manifest().values()}

    assert owned & shared == set(), f'records declared twice: {sorted(owned & shared)}'
    # A key in both would make the PROTEUS entry shadow the shared one.
    assert {ds.key for ds in owned_datasets} & set(_shared_manifest()) == set()
    # Discrimination: both sets are populated, and the shared one holds records
    # PROTEUS reads, so the empty intersection is not from an empty load.
    assert len(owned) == len(_OWNED_KEYS)
    assert f'10.5281/zenodo.{PHOENIX_RECORD}' in shared


def test_registries_pin_committed_checksums():
    """Each dataset PROTEUS reads has a registry with the files PROTEUS fetches.

    The counts and one literal digest are pinned so a truncated or regenerated
    registry is caught; an empty registry would otherwise verify nothing while
    still loading cleanly. For shared datasets the files PROTEUS fetches one by
    one must be listed, or the per-file fetch raises KeyError.
    """
    exo = _dataset(EXOPLANET_REFERENCE).registry()
    zeng = _dataset(MASS_RADIUS_ZENG_2019).registry()
    hammond = _dataset(SURFACE_ALBEDOS_HAMMOND_2024).registry()
    seager = _dataset(EOS_SEAGER_2007).registry()
    solar = _dataset(STELLAR_SPECTRA_SOLAR).registry()
    named = _dataset(STELLAR_SPECTRA_NAMED).registry()
    muscles = _dataset(STELLAR_SPECTRA_MUSCLES).registry()
    wolf_bower = _dataset(EOS_WOLF_BOWER_2018).registry()
    rtpress = _dataset(EOS_RTPRESS_100TPA).registry()
    paleos_2phase = _dataset(EOS_PALEOS_MGSIO3).registry()
    chabrier = _dataset(EOS_CHABRIER_2021).registry()
    lookup = _dataset(LOOKUP_WOLF_BOWER_2018_1TPA).registry()
    monteux_plus600 = _dataset(MELTING_MONTEUX_PLUS600).registry()
    monteux_minus600 = _dataset(MELTING_MONTEUX_MINUS600).registry()
    melting_wolf_bower = _dataset(MELTING_WOLF_BOWER_2018).registry()

    assert len(exo) == 1, 'the catalogue ships exactly one file'
    assert len(zeng) == 57, 'the Zeng-2019 grid ships 57 curve files'
    assert len(hammond) == 26, 'the Hammond-2024 record ships 25 spectra and a readme'
    assert set(seager) == {
        'eos_seager07_iron.txt',
        'eos_seager07_silicate.txt',
        'eos_seager07_water.txt',
    }
    assert len(solar) == 10, 'the solar record ships 10 spectra'
    assert len(named) == 11, 'the named-star record ships 11 spectra'
    assert len(muscles) == 38, 'the MUSCLES record ships 36 spectra, a readme and a table'
    assert {'density_melt.dat', 'density_solid.dat', 'adiabat_temp_grad_melt.dat'} <= set(
        wolf_bower
    )
    assert {'density_melt.dat', 'adiabat_temp_grad_melt.dat'} <= set(rtpress)
    assert set(paleos_2phase) == {
        'paleos_mgsio3_tables_pt_proteus_liquid.dat',
        'paleos_mgsio3_tables_pt_proteus_solid.dat',
        'paleos_mgsio3_tables_pt_proteus_liquid_highres.dat',
        'paleos_mgsio3_tables_pt_proteus_solid_highres.dat',
    }
    # The unified record also holds high-resolution variants (2.29 GB in all);
    # each material's dataset lists only its own table.
    assert set(_dataset(EOS_PALEOS_IRON).registry()) == {'paleos_iron_eos_table_pt.dat'}
    assert set(_dataset(EOS_PALEOS_MGSIO3_UNIFIED).registry()) == {
        'paleos_mgsio3_eos_table_pt.dat'
    }
    assert set(_dataset(EOS_PALEOS_H2O).registry()) == {'paleos_water_eos_table_pt.dat'}
    assert set(chabrier) == {'EOS_Chabrier2021_HHe.tar.gz'}
    assert chabrier['EOS_Chabrier2021_HHe.tar.gz'] == 'md5:18ce96ed0526d4ade283807a7da2e091'
    assert len(lookup) == 14, 'the Wolf-Bower lookup-table record ships 14 files'
    assert set(monteux_plus600) == {'liquidus.dat', 'solidus.dat'}
    assert set(monteux_minus600) == {'liquidus.dat', 'solidus.dat'}
    assert set(melting_wolf_bower) == {'liquidus.dat', 'solidus.dat'}
    assert lookup['thermal_exp_melt.dat'] == 'md5:f2983441e654c4d4285f9a0e563247da'
    assert lookup['density_melt.dat'] == 'md5:8aaa37a5ac723e248895e0e6eb609046'
    assert monteux_plus600['solidus.dat'] == 'md5:e70660c7ff400609fc3ed442ea0eac84'
    assert monteux_minus600['solidus.dat'] == 'md5:6d48d841639c06f61a872656fc9a5b11'
    assert melting_wolf_bower['solidus.dat'] == 'md5:66b297a120c79e7e58de5761ac40f4c4'
    assert solar['sun.txt'] == 'md5:6e4b6540d952cf3c01a3bd0511aa6c10'
    assert named['sun.txt'] == 'md5:0c5225b847ca250673edc9690f02e055'
    assert muscles['gj876.txt'] == 'md5:4b4e7299bad9545ee8fef82cb4c8d4d8'
    assert seager['eos_seager07_iron.txt'] == 'md5:7bf215a2bb4da6d27ceeac2ade0ce706'
    assert hammond['lunarmarebasalt.dat'] == 'md5:a157ea1d436072264c3bea833997a382'
    assert exo['DACE_PlanetS.csv'] == 'md5:367a90914eba4a209f896a1c72dd3d2b'
    # Every entry must carry an algorithm prefix, or pooch cannot know what to
    # verify against; a bare digest would silently be read as the default.
    for registry in (exo, zeng, hammond, seager):
        assert all(':' in digest for digest in registry.values())
    assert 'massradiusEarthlikeRocky.txt' in zeng


def test_manifest_is_discovered_via_entry_point():
    """fwl-io finds PROTEUS's datasets through the installed entry point.

    This is what makes ``fwl-io fetch proteus`` work without importing PROTEUS,
    so a typo in the entry-point name or target is caught here rather than by a
    user with an empty data tree.
    """
    from fwl_io import discover_manifests

    providers = discover_manifests()

    assert 'proteus' in providers, f'proteus not among providers: {sorted(providers)}'
    assert {ds.key for ds in providers['proteus']} == _OWNED_KEYS


def test_dataset_dir_is_versioned(tmp_path):
    """A dataset resolves into the version directory named for its record.

    The literal path is pinned because the version segment is what keeps a
    re-pinned deposit from overwriting its predecessor. The discrimination
    assert rules out the bare location, which is one directory above where the
    readers look and is exactly what a version-less resolution would return.
    """
    resolved = dataset_dir(MASS_RADIUS_ZENG_2019, data_root=tmp_path)

    assert (
        resolved == tmp_path / 'observe' / 'mass_radius' / 'zeng_2019' / f'r{ZENG_2019_RECORD}'
    )
    assert resolved != tmp_path / 'observe' / 'mass_radius' / 'zeng_2019'
    assert dataset_dir(EXOPLANET_REFERENCE, data_root=tmp_path) == (
        tmp_path / 'observe' / 'exoplanet_reference' / f'r{EXOPLANET_RECORD}'
    )
    assert dataset_dir(SURFACE_ALBEDOS_HAMMOND_2024, data_root=tmp_path) == (
        tmp_path / 'atmos_clim' / 'surface_albedos' / 'hammond_2024' / f'r{HAMMOND_2024_RECORD}'
    )
    assert dataset_dir(EOS_SEAGER_2007, data_root=tmp_path) == (
        tmp_path / 'interior_struct' / 'eos' / 'seager_2007' / f'r{SEAGER_2007_RECORD}'
    )
    for key, (subdir, record) in SHARED_DATASETS.items():
        assert dataset_dir(key, data_root=tmp_path) == tmp_path / subdir / f'r{record}'
    assert dataset_dir(STELLAR_SPECTRA_PHOENIX, data_root=tmp_path) == (
        tmp_path / 'star' / 'spectra' / 'phoenix' / f'r{PHOENIX_RECORD}'
    )


def test_dataset_dir_rejects_an_unversioned_resolution(tmp_path, monkeypatch):
    """An unversioned resolution fails loudly instead of returning a wrong path.

    Serving the bare location would put every reader one directory above the
    files and look like missing data, so the guard names the offending path.
    A manifest dataset always carries a version DOI, so this branch is not
    reachable through the shipped manifest; it is a tripwire for a future fwl-io
    that resolves a location without a version segment, and is driven here
    through a stand-in fetcher.
    """

    class _UnversionedFetcher:
        version_dir = None
        target_dir = tmp_path / 'observe' / 'mass_radius' / 'zeng_2019'

    monkeypatch.setattr('proteus.data._fetcher', lambda *a, **k: _UnversionedFetcher())

    with pytest.raises(RuntimeError, match='unversioned') as raised:
        dataset_dir(MASS_RADIUS_ZENG_2019, data_root=tmp_path)

    # Discrimination: the guard names the path it refused, which is what sends
    # the reader to the right directory. A message saying only that something
    # was unversioned would satisfy the match above and tell them nothing.
    assert 'zeng_2019' in str(raised.value)


def test_unknown_dataset_key_is_rejected():
    """A key absent from both manifests raises rather than resolving somewhere.

    Silently resolving an undeclared key would create an unpinned directory with
    no registry to verify against.
    """
    with pytest.raises(KeyError, match=f'fwl-io>={FWL_IO_FLOOR}'):
        _dataset('observe.not_a_declared_dataset')

    # Discrimination: a key the manifest does declare resolves, and to that
    # same dataset, so the raise above follows from the key being absent
    # rather than from the lookup refusing or mis-resolving what it is asked.
    assert _dataset(MASS_RADIUS_ZENG_2019).key == MASS_RADIUS_ZENG_2019


def test_stale_fwl_io_is_named_as_the_stale_side(monkeypatch):
    """An fwl-io that predates the manifest schema is reported as out of date.

    Such an fwl-io rejects the shipped manifest as malformed, which would point
    the reader at a file they must not edit; the error has to name the installed
    package as the stale side instead.
    """
    import fwl_io

    def _rejects(*args, **kwargs):
        raise ValueError('"subdir" is not a manifest field')

    monkeypatch.setattr(fwl_io, 'load_manifest', _rejects)
    monkeypatch.setattr('proteus.data._fwl_io_derives_the_location', lambda: False)

    with pytest.raises(RuntimeError, match=f'upgrade to fwl-io>={FWL_IO_FLOOR}') as excinfo:
        _dataset(MASS_RADIUS_ZENG_2019)

    assert isinstance(excinfo.value.__cause__, ValueError), 'the original error stays attached'


def test_manifest_error_under_a_current_fwl_io_propagates(monkeypatch):
    """A real defect in the shipped manifest surfaces as itself, not as a version claim.

    Blaming a current fwl-io for a manifest we ship would send the reader to fix
    the wrong thing.
    """
    import fwl_io

    def _rejects(*args, **kwargs):
        raise ValueError('dataset key is malformed')

    monkeypatch.setattr(fwl_io, 'load_manifest', _rejects)
    monkeypatch.setattr('proteus.data._fwl_io_derives_the_location', lambda: True)

    with pytest.raises(ValueError, match='malformed') as raised:
        _dataset(MASS_RADIUS_ZENG_2019)

    # Discrimination: the message carries the manifest's own complaint and not
    # an upgrade instruction, which is the whole point of separating a defect
    # we ship from a dependency that is out of date.
    assert 'fwl-io' not in str(raised.value).lower()


def test_capability_check_reads_the_installed_fwl_io(monkeypatch):
    """The staleness check is answered from the installed dataset fields.

    Reading the fields rather than a version string means the check follows the
    schema itself; an fwl-io that cannot be introspected counts as current, so a
    version mismatch is never asserted without evidence.
    """
    import fwl_io.manifest

    @dataclasses.dataclass
    class _OldDataset:
        key: str = ''
        subdir: str = ''

    @dataclasses.dataclass
    class _NewDataset:
        key: str = ''

    monkeypatch.setattr(fwl_io.manifest, 'Dataset', _OldDataset)
    assert _fwl_io_derives_the_location() is False

    monkeypatch.setattr(fwl_io.manifest, 'Dataset', _NewDataset)
    assert _fwl_io_derives_the_location() is True


def test_declared_floor_is_not_below_the_schema_floor():
    """The pyproject fwl-io floor is not below the manifest schema floor.

    A pyproject floor below the schema floor would let pip install an fwl-io that
    cannot read the manifest, which the load reports as a stale install. A
    pyproject floor above it is allowed: it tracks fixes in later fwl-io
    releases, and the upgrade instruction names only the schema floor.
    """
    from packaging.requirements import Requirement
    from packaging.version import Version

    requirements = _pyproject()['project']['dependencies']
    bounds = [
        spec.version
        for req in (Requirement(text) for text in requirements)
        if req.name == 'fwl-io'
        for spec in req.specifier
        if spec.operator == '>='
    ]

    # Checked first so a dropped `>=` reports the absence it is, rather than
    # reaching the comparison below and reading as a version mismatch.
    assert len(bounds) == 1, f'expected one lower bound on fwl-io, found {bounds}'
    assert Version(bounds[0]) >= Version(FWL_IO_FLOOR), (
        f'pyproject floor {bounds[0]} is below the manifest schema floor {FWL_IO_FLOOR}'
    )


def test_manifest_and_registries_are_declared_as_package_data():
    """The manifest and its registries are declared as package data.

    They are read from the installed package, so leaving them out of the wheel
    works in a source checkout and fails only on a user's machine. This checks
    the declaration and the shipped filenames; that a built wheel really carries
    them is covered at the integration tier.
    """
    package_data = _pyproject()['tool']['setuptools']['package-data']

    assert 'proteus.data' in package_data, 'proteus.data declares no package data'
    patterns = package_data['proteus.data']
    assert '*.toml' in patterns
    assert '*.registry.txt' in patterns
    # The declared patterns must actually match the shipped files, not merely
    # exist: a rename of the manifest would satisfy the globs above alone.
    shipped = {path.name for path in manifest_path().parent.iterdir() if path.is_file()}
    assert 'proteus_manifest.toml' in shipped
    assert {name for name in shipped if name.endswith('.registry.txt')} == {
        f'{key}.registry.txt' for key in _OWNED_KEYS
    }


def test_spectral_file_datasets_pin_their_records_and_locations(tmp_path):
    """Each (group, bands) spectral file set is its own dataset at a derived path.

    The record ids are repeated literally so a silent re-pin fails here, and the
    resolved directory is pinned because AGNI and JANUS read the file from it.
    """
    for (group, bands), record in SPECTRAL_RECORDS.items():
        key = spectral_file_key(group, bands)
        dataset = _dataset(key)
        assert dataset.zenodo == f'10.5281/zenodo.{record}'
        assert dataset.subdir == f'atmos_clim/spectral_files/{group.lower()}/{bands}'
        assert dataset_dir(key, data_root=tmp_path) == (
            tmp_path / 'atmos_clim' / 'spectral_files' / group.lower() / bands / f'r{record}'
        )
        assert _dataset(key).registry(), f'empty registry for {key}'


def test_every_spectral_folder_has_a_dataset_and_no_legacy_entry():
    """The download list and the manifest agree, and the OSF-era map holds none.

    A folder without a manifest table would fail only when a user runs the
    download; a leftover map entry would give the record a second pin.
    """
    from proteus.utils.data import DATA_SOURCE_MAP, SPECTRAL_FILE_FOLDERS

    assert {tuple(folder.split('/')) for folder in SPECTRAL_FILE_FOLDERS} == set(
        SPECTRAL_RECORDS
    )
    for folder in SPECTRAL_FILE_FOLDERS:
        group, bands = folder.split('/')
        assert _dataset(spectral_file_key(group, bands)).registry()
        assert folder not in DATA_SOURCE_MAP, f'{folder} is still pinned in DATA_SOURCE_MAP'


def test_get_spfile_path_resolves_into_the_versioned_dataset_dir(tmp_path):
    """The path PROTEUS hands to AGNI and JANUS is inside the dataset's version dir."""
    from types import SimpleNamespace

    from proteus.atmos_clim.common import get_spfile_path

    config = SimpleNamespace(
        atmos_clim=SimpleNamespace(module='agni', spectral_group='Oak', spectral_bands='318')
    )
    path = Path(get_spfile_path(str(tmp_path), config))

    assert path == dataset_dir(spectral_file_key('Oak', '318'), data_root=tmp_path) / 'Oak.sf'
    assert path.parent.name == 'r15743843'


def test_unknown_spectral_pair_is_rejected():
    """An unlisted (group, bands) pair fails loudly instead of resolving to a path."""
    key = spectral_file_key('Oak', '16')

    assert key == 'atmos_clim.spectral_files.oak.16'
    with pytest.raises(KeyError, match=r'oak\.16'):
        dataset_dir(key)


def test_migrated_datasets_are_not_also_pinned_in_the_legacy_map():
    """A migrated dataset is pinned in one place only.

    Leaving its record in the legacy mapping too would let the two pins drift,
    so a re-pin of the manifest would silently keep fetching the old deposit
    through whichever path ran first.
    """
    from proteus.utils.data import DATA_SOURCE_MAP

    assert 'Exoplanets' not in DATA_SOURCE_MAP
    assert 'Zeng2019' not in DATA_SOURCE_MAP
    assert 'Hammond24' not in DATA_SOURCE_MAP
    assert 'EOS_Seager2007' not in DATA_SOURCE_MAP
    for legacy in (
        'EOS_WolfBower2018_1TPa',
        'EOS_RTPress_melt_100TPa',
        'EOS_PALEOS_MgSiO3',
        'EOS_PALEOS_iron',
        'EOS_PALEOS_MgSiO3_unified',
        'EOS_PALEOS_H2O',
        'EOS_Chabrier2021_HHe',
    ):
        assert legacy not in DATA_SOURCE_MAP
    assert 'Population' not in DATA_SOURCE_MAP
    # Discrimination: the map is still populated for the datasets that have not
    # migrated, so an emptied map cannot make this pass.
    assert 'Named' not in DATA_SOURCE_MAP
    assert 'solar' not in DATA_SOURCE_MAP
    assert 'MUSCLES' not in DATA_SOURCE_MAP
    assert 'PHOENIX' not in DATA_SOURCE_MAP
    assert 'scattering' in DATA_SOURCE_MAP
    pinned_records = {entry['zenodo_id'] for entry in DATA_SOURCE_MAP.values()}
    assert EXOPLANET_RECORD not in pinned_records
    assert HAMMOND_2024_RECORD not in pinned_records
    assert SEAGER_2007_RECORD not in pinned_records
    assert ZENG_2019_RECORD not in pinned_records
    assert SOLAR_RECORD not in pinned_records
    assert NAMED_RECORD not in pinned_records
    assert MUSCLES_RECORD not in pinned_records
    assert PHOENIX_RECORD not in pinned_records


def test_fetch_dataset_delegates_to_the_pinned_fetcher(monkeypatch, tmp_path):
    """Fetching a dataset runs the fetcher built from the manifest pin.

    The fetch is delegated rather than reimplemented, so this pins that the
    dataset key reaches the fetcher and that no download happens here.
    """
    calls = {'fetch_all': 0, 'key': None, 'data_root': 'unset'}

    class _Fetcher:
        def fetch_all(self):
            calls['fetch_all'] += 1
            return [tmp_path / 'a.txt']

    def _fake_fetcher(key, data_root=None):
        calls['key'] = key
        calls['data_root'] = data_root
        return _Fetcher()

    monkeypatch.setattr('proteus.data._fetcher', _fake_fetcher)

    result = fetch_dataset(MASS_RADIUS_ZENG_2019, data_root=tmp_path)

    assert calls['fetch_all'] == 1, 'the fetch runs exactly once'
    assert calls['key'] == MASS_RADIUS_ZENG_2019
    # The caller-supplied tree has to reach the fetcher, or the fetch would
    # silently populate the process-wide data root instead.
    assert calls['data_root'] == tmp_path
    assert result == [tmp_path / 'a.txt']


def test_default_data_root_is_the_tree_proteus_reads_from(monkeypatch, tmp_path):
    """A dataset fetched with no explicit root lands below the run's data root.

    The fetch side resolves the root from the module-level constant frozen at
    import; a resolution that ignored it would populate a different tree from
    the one the run reads.
    """
    monkeypatch.setenv('FWL_DATA', str(tmp_path))
    monkeypatch.setattr('proteus.utils.data.FWL_DATA_DIR', tmp_path, raising=False)

    implicit = dataset_dir(MASS_RADIUS_ZENG_2019)
    explicit = dataset_dir(MASS_RADIUS_ZENG_2019, data_root=tmp_path)

    assert implicit == explicit
    # Discrimination: a resolution that fell back to the repo-local default
    # would still be a valid path, but not one below the run's data root.
    assert implicit.is_relative_to(tmp_path)


def test_fetch_and_read_sides_agree_on_a_home_relative_data_root(monkeypatch, tmp_path):
    """PROTEUS and fwl-io must resolve ``FWL_DATA`` to the same tree.

    Both resolvers run in this path: PROTEUS resolves the root for the fetch,
    fwl-io resolves the root the readers are handed. A value carrying a literal
    '~' is where they can disagree, and a disagreement means ``proteus get
    reference`` reports success while the population diagram warns that the
    data is missing. Compares the two implementations directly rather than
    asserting either one in isolation.
    """
    from fwl_io.paths import resolve_data_root

    from proteus.utils.helper import resolve_fwl_data_dir

    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('FWL_DATA', '~/fwl_data_root_check')

    proteus_side = Path(resolve_fwl_data_dir()).absolute()
    fwl_io_side = resolve_data_root()

    assert proteus_side == fwl_io_side
    # Discrimination: the failure this guards is a literal '~' directory below
    # the working directory, which is still an absolute, plausible-looking path.
    assert '~' not in str(proteus_side)
    assert proteus_side.is_relative_to(tmp_path)
