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
    EXOPLANET_REFERENCE,
    FWL_IO_FLOOR,
    MASS_RADIUS_ZENG_2019,
    _dataset,
    _fwl_io_derives_the_location,
    dataset_dir,
    fetch_dataset,
    manifest_path,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# The pinned records, repeated here so a silent re-pin of the manifest fails a
# test rather than quietly moving every reader to a different deposit.
EXOPLANET_RECORD = '15727878'
ZENG_2019_RECORD = '15727899'

REPO_ROOT = Path(__file__).resolve().parents[2]


def _pyproject() -> dict:
    """Return the parsed pyproject.toml of the repository under test."""
    with open(REPO_ROOT / 'pyproject.toml', 'rb') as handle:
        return tomllib.load(handle)


def test_manifest_declares_the_observe_datasets():
    """The manifest declares the two reference datasets PROTEUS owns.

    Both the key set and each dataset's derived location are pinned: fwl-io
    turns the dotted key into the on-disk path, so a key edit silently relocates
    the data unless the mapping is asserted.
    """
    from fwl_io import load_manifest

    datasets = {ds.key: ds for ds in load_manifest(manifest_path())}

    assert set(datasets) == {EXOPLANET_REFERENCE, MASS_RADIUS_ZENG_2019}
    assert datasets[EXOPLANET_REFERENCE].subdir == 'observe/exoplanet_reference'
    assert datasets[MASS_RADIUS_ZENG_2019].subdir == 'observe/mass_radius/zeng_2019'
    assert datasets[EXOPLANET_REFERENCE].zenodo == f'10.5281/zenodo.{EXOPLANET_RECORD}'
    assert datasets[MASS_RADIUS_ZENG_2019].zenodo == f'10.5281/zenodo.{ZENG_2019_RECORD}'
    # Both are PROTEUS-owned, so "proteus" has to appear in required_by or
    # "fwl-io fetch proteus" would skip them.
    for ds in datasets.values():
        assert 'proteus' in [model.lower() for model in ds.required_by]


def test_registries_pin_committed_checksums():
    """Each dataset ships a registry of file checksums generated from its record.

    The counts and one literal digest are pinned so a truncated or regenerated
    registry is caught; an empty registry would otherwise verify nothing while
    still loading cleanly.
    """
    exo = _dataset(EXOPLANET_REFERENCE).registry()
    zeng = _dataset(MASS_RADIUS_ZENG_2019).registry()

    assert len(exo) == 1, 'the catalogue ships exactly one file'
    assert len(zeng) == 57, 'the Zeng-2019 grid ships 57 curve files'
    assert exo['DACE_PlanetS.csv'] == 'md5:367a90914eba4a209f896a1c72dd3d2b'
    # Every entry must carry an algorithm prefix, or pooch cannot know what to
    # verify against; a bare digest would silently be read as the default.
    for registry in (exo, zeng):
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
    assert {ds.key for ds in providers['proteus']} == {
        EXOPLANET_REFERENCE,
        MASS_RADIUS_ZENG_2019,
    }


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

    with pytest.raises(RuntimeError, match='unversioned'):
        dataset_dir(MASS_RADIUS_ZENG_2019, data_root=tmp_path)


def test_unknown_dataset_key_is_rejected():
    """A key absent from the manifest raises rather than resolving somewhere.

    Silently resolving an undeclared key would create an unpinned directory with
    no registry to verify against.
    """
    with pytest.raises(KeyError):
        _dataset('observe.not_a_declared_dataset')


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

    with pytest.raises(ValueError, match='malformed'):
        _dataset(MASS_RADIUS_ZENG_2019)


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


def test_declared_floor_matches_the_requirement():
    """The fwl-io floor in the error message equals the one pip installs.

    If the two drift, the upgrade instruction names a version that does not fix
    the problem the reader is looking at.
    """
    from packaging.requirements import Requirement

    requirements = _pyproject()['project']['dependencies']
    bounds = [
        spec.version
        for req in (Requirement(text) for text in requirements)
        if req.name == 'fwl-io'
        for spec in req.specifier
        if spec.operator == '>='
    ]

    assert bounds == [FWL_IO_FLOOR], f'pyproject floor {bounds} vs module floor {FWL_IO_FLOOR}'


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
        f'{EXOPLANET_REFERENCE}.registry.txt',
        f'{MASS_RADIUS_ZENG_2019}.registry.txt',
    }


def test_migrated_datasets_are_not_also_pinned_in_the_legacy_map():
    """A migrated dataset is pinned in one place only.

    Leaving its record in the legacy mapping too would let the two pins drift,
    so a re-pin of the manifest would silently keep fetching the old deposit
    through whichever path ran first.
    """
    from proteus.utils.data import DATA_SOURCE_MAP

    assert 'Exoplanets' not in DATA_SOURCE_MAP
    assert 'Zeng2019' not in DATA_SOURCE_MAP
    # Discrimination: the map is still populated for the datasets that have not
    # migrated, so an emptied map cannot make this pass.
    assert 'Hammond24' in DATA_SOURCE_MAP
    pinned_records = {entry['zenodo_id'] for entry in DATA_SOURCE_MAP.values()}
    assert EXOPLANET_RECORD not in pinned_records
    assert ZENG_2019_RECORD not in pinned_records


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
