"""Unit tests for ``proteus.utils.prt_data``.

Covers the reference-data plumbing behind the optional observe module:

- the opacity tables PROTEUS asks for are named explicitly, and completeness is judged
  against that set rather than against whatever is in the directory
- a partly-fetched tree is not mistaken for a complete one
- an interrupted transfer leaves no truncated table behind
- the petitRADTRANS configuration names one file per species, which is what keeps the
  library from stopping to ask which to use
- writing a configuration against an incomplete tree is refused

The module is structural plumbing (path resolution, HTTP fetching, config writing) and
computes no physical quantity, so the physics-invariant requirement does not apply.

Testing standards:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

import re

import pytest

from proteus.utils import prt_data
from proteus.utils.constants import prt_cia_species

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# A stand-in for the named table set, small enough to reason about and shaped like the
# real one: two line species under different directories, plus a collision pair.
FAKE_FILES = {
    'opacities/lines/correlated_k/H2O/1H2-16O': 'water.ktable.petitRADTRANS.h5',
    'opacities/lines/correlated_k/CO2/12C-16O2': 'co2.ktable.petitRADTRANS.h5',
    'opacities/continuum/collision_induced_absorptions/H2--He/H2--He-NatAbund': (
        'h2he.ciatable.petitRADTRANS.h5'
    ),
}


@pytest.fixture
def tree(monkeypatch, tmp_path):
    """Point the module at a temporary data directory with a known table set."""
    monkeypatch.setattr(prt_data, 'GetFWLData', lambda: tmp_path)
    monkeypatch.setattr(prt_data, 'PRT_DEFAULT_FILES', dict(FAKE_FILES))
    monkeypatch.setattr(prt_data, '_config_path', lambda: tmp_path / 'cfg' / 'prt.ini')
    return tmp_path / 'prt' / 'input_data'


def _place(root, sub):
    """Put one named table on disk."""
    dest = root / sub / FAKE_FILES[sub]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b'\x89HDF\r\n\x1a\n' + b'x' * 8)
    return dest


def _raise(exc):
    """Return a stand-in transfer that fails."""

    def fail(*_args, **_kwargs):
        raise exc

    return fail


class _FakeResponse:
    """Stand-in for a urlopen response, so the real transfer can be driven."""

    def __init__(self, body, content_type='application/octet-stream', fail_after=None):
        self._body = body
        self._pos = 0
        self._fail_after = fail_after
        self.headers = {'Content-Type': content_type}

    def read(self, size=-1):
        if self._fail_after is not None and self._pos >= self._fail_after:
            raise OSError('connection reset mid-transfer')
        chunk = self._body[
            self._pos : self._pos + (size if size and size > 0 else len(self._body))
        ]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _serve(response):
    """Patch urlopen to return one prepared response."""

    def opener(*_args, **_kwargs):
        return response

    return opener


def test_completeness_is_judged_against_the_named_tables(tree):
    """A tree counts as present only when every named table is there, not when some are.

    Judging by "is any table on disk" would call a run that stopped after the first table
    complete, and nothing would then fetch the rest: petitRADTRANS would go looking for
    them over the network mid-run. Two of the three tables are placed to make the
    difference visible, since a check that answered for the directory as a whole would
    pass at that point.
    """
    assert prt_data.opacities_present() is False

    _place(tree, 'opacities/lines/correlated_k/H2O/1H2-16O')
    _place(tree, 'opacities/lines/correlated_k/CO2/12C-16O2')
    assert prt_data.opacities_present() is False, 'a partly-fetched tree is not complete'
    assert [p.name for p in prt_data.missing_tables()] == ['h2he.ciatable.petitRADTRANS.h5']

    _place(tree, 'opacities/continuum/collision_induced_absorptions/H2--He/H2--He-NatAbund')
    assert prt_data.opacities_present() is True
    assert prt_data.missing_tables() == []


def test_only_the_absent_tables_are_fetched(tree, monkeypatch):
    """A fetch collects what is missing and leaves what is already there.

    These tables run to gigabytes, so re-fetching a present one is not a harmless
    inefficiency. The already-present table is asserted untouched rather than only
    checking the outcome, because a fetch that downloaded everything and then reported
    success would satisfy an outcome check while costing the whole transfer again.
    """
    _place(tree, 'opacities/lines/correlated_k/H2O/1H2-16O')

    fetched = []

    def fake_download(library_path, destination):
        fetched.append(library_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b'\x89HDF\r\n\x1a\n')

    monkeypatch.setattr(prt_data, '_download', fake_download)

    assert prt_data.download_prt_opacities() is True
    assert sorted(fetched) == [
        '/opacities/continuum/collision_induced_absorptions/H2--He/H2--He-NatAbund/h2he.ciatable.petitRADTRANS.h5',
        '/opacities/lines/correlated_k/CO2/12C-16O2/co2.ktable.petitRADTRANS.h5',
    ], 'the table already on disk should not have been fetched again'
    assert prt_data.opacities_present() is True


def test_a_complete_tree_is_left_alone(tree, monkeypatch):
    """Nothing is fetched when every table is already in place."""
    for sub in FAKE_FILES:
        _place(tree, sub)

    def refuse(*_a, **_k):
        raise AssertionError('fetch attempted for a tree that is already complete')

    monkeypatch.setattr(prt_data, '_download', refuse)

    assert prt_data.download_prt_opacities() is True
    # The configuration is written even so: the tables are useless if the library cannot
    # be told where they are, and a machine may have the tables from an earlier run.
    assert (tree.parent.parent / 'cfg' / 'prt.ini').is_file()


def test_clean_refetches_tables_that_are_already_present(tree, monkeypatch):
    """Asking for a clean fetch collects every table again, not just the absent ones.

    This is the path for replacing a tree that is present but suspect, so skipping the
    tables already on disk would make the option do nothing.
    """
    _place(tree, 'opacities/lines/correlated_k/H2O/1H2-16O')

    fetched = []
    monkeypatch.setattr(
        prt_data,
        '_download',
        lambda lib, dest: (
            fetched.append(lib),
            dest.parent.mkdir(parents=True, exist_ok=True),
            dest.write_bytes(b'\x89HDF\r\n\x1a\n'),
        ),
    )

    assert prt_data.download_prt_opacities(clean=True) is True
    assert len(fetched) == len(FAKE_FILES)


def test_a_failed_transfer_is_reported_to_the_caller(tree, monkeypatch):
    """A transfer that raises stops the fetch and reports failure.

    The caller decides from this answer whether synthesis can go ahead, so the failure
    has to reach it rather than be logged and forgotten. This covers the reporting
    contract only; what the transfer itself leaves on disk is checked against the real
    function below, since a stand-in cannot demonstrate the real one's cleanup.
    """
    monkeypatch.setattr(prt_data, '_download', _raise(OSError('connection reset')))

    assert prt_data.download_prt_opacities() is False
    assert prt_data.opacities_present() is False


def test_a_transfer_that_delivers_nothing_is_reported_as_failure(tree, monkeypatch):
    """A fetch that claims success while writing no table reports failure.

    Returning success over an empty directory would push the problem into petitRADTRANS,
    where it surfaces as a download attempt mid-run instead of a plain message here.
    """
    monkeypatch.setattr(prt_data, '_download', lambda lib, dest: None)

    assert prt_data.download_prt_opacities() is False


def test_config_names_one_file_per_species_under_the_right_headings(tree):
    """The configuration records the data path and names a file for every species.

    A configuration carrying the path but no names would load, and petitRADTRANS would
    then stop to ask on the first species offering more than one table, so both halves
    are checked. The names must also precede the [Paths] heading: this is read as an ini
    file, and a name landing in the wrong section is ignored without complaint.
    """
    for sub in FAKE_FILES:
        _place(tree, sub)

    text = prt_data.write_prt_config().read_text()
    lines = text.splitlines()

    assert lines[0] == '[Default files]'
    assert f'prt_input_data_path = {tree}' in lines

    named = [line for line in lines if line.startswith('opacities/')]
    assert len(named) == len(FAKE_FILES)
    assert 'opacities/lines/correlated_k/H2O/1H2-16O = water.ktable.petitRADTRANS.h5' in named
    assert max(lines.index(n) for n in named) < lines.index('[Paths]')


def test_config_is_refused_while_any_table_is_missing(tree):
    """Writing a configuration against an incomplete tree is refused, and says which.

    Naming a file that is not on disk is how petitRADTRANS ends up fetching it mid-run,
    which is the behaviour this module exists to prevent. The message names the directory
    and one absent table, because the usual causes are a data directory pointing
    somewhere unexpected and a fetch that did not finish.
    """
    _place(tree, 'opacities/lines/correlated_k/H2O/1H2-16O')

    with pytest.raises(FileNotFoundError, match=re.escape(str(tree))) as excinfo:
        prt_data.write_prt_config()
    assert 'co2.ktable.petitRADTRANS.h5' in str(excinfo.value) or 'h2he' in str(excinfo.value)

    for sub in FAKE_FILES:
        _place(tree, sub)
    assert prt_data.write_prt_config().is_file()


def test_every_gas_proteus_models_has_a_named_table(monkeypatch):
    """The shipped table set covers every gas the observe module may ask for.

    A gas with no named table is dropped from the transfer and its opacity is silently
    absent from the spectrum, so the two lists are checked against each other rather than
    left to drift apart. This reads the real constants, not the fixture, because the
    point is the shipped set.
    """
    assert prt_data.uncovered_species() == []

    # A gas added to the model but not to the table set is reported, not hidden.
    monkeypatch.setattr(prt_data, 'PRT_LINE_SPECIES', prt_data.PRT_LINE_SPECIES + ('XeF6',))
    assert prt_data.uncovered_species() == ['XeF6']


def test_named_tables_sit_where_petitradtrans_looks_for_them():
    """Every named table is keyed by a path petitRADTRANS recognises.

    petitRADTRANS reads line opacities and collision-induced absorption from two separate
    roots, and a name filed under the wrong one is silently ignored: the species then
    resolves to a prompt at run time. Checking the keys catches that here rather than on
    an unattended machine.
    """
    line_root = 'opacities/lines/correlated_k/'
    cia_root = 'opacities/continuum/collision_induced_absorptions/'

    for sub, name in prt_data.PRT_DEFAULT_FILES.items():
        assert sub.startswith((line_root, cia_root)), f'unrecognised location: {sub}'
        # Species directory plus isotopologue directory, which is the depth the library
        # stores tables at and the depth the config keys off.
        assert len(sub.removeprefix(line_root).removeprefix(cia_root).split('/')) == 2, sub
        assert name.endswith('.petitRADTRANS.h5'), f'not a petitRADTRANS table: {name}'

    # Every collision pair the observe module asks for is named, and none beyond them.
    cia_named = {
        sub.split('/')[3] for sub in prt_data.PRT_DEFAULT_FILES if sub.startswith(cia_root)
    }
    assert cia_named == set(prt_cia_species)


def test_the_transfer_rejects_a_web_page_served_in_place_of_a_table(tree, monkeypatch):
    """A web page returned where a table was asked for is refused, not saved.

    This is the shape of an upstream rename: the library answers a path it no longer holds
    with a page and a success status, so nothing raises on its own and the staging rename
    offers no protection. Saved unchecked, the page would sit at the table's name, satisfy
    every completeness check, be named in the configuration, and finally fail inside
    petitRADTRANS while parsing it, a long way from the cause. Both the declared type and
    the opening bytes are checked, since a library that mislabels its response would slip
    past the first on its own.
    """
    page = b'\n\n<!DOCTYPE html>\n<html>KEEPER</html>'
    dest = (
        tree
        / 'opacities/lines/correlated_k/H2O/1H2-16O'
        / FAKE_FILES['opacities/lines/correlated_k/H2O/1H2-16O']
    )

    monkeypatch.setattr(
        prt_data.urllib.request,
        'urlopen',
        _serve(_FakeResponse(page, content_type='text/html')),
    )
    with pytest.raises(OSError, match='web page'):
        prt_data._download('/some/renamed/path.h5', dest)
    assert not dest.exists()

    # Same page, but the library calls it a table: the opening bytes still give it away.
    monkeypatch.setattr(
        prt_data.urllib.request,
        'urlopen',
        _serve(_FakeResponse(page, content_type='application/octet-stream')),
    )
    with pytest.raises(OSError, match='not an HDF5 table'):
        prt_data._download('/some/renamed/path.h5', dest)
    assert not dest.exists()
    assert list(tree.rglob('*.part')) == []


def test_the_transfer_stages_the_table_and_clears_up_after_itself(tree, monkeypatch):
    """A table arrives at its final name only once it is complete, and a transfer that
    dies part-way leaves neither a staged file nor a truncated table.

    A half-written table is worse than an absent one: it satisfies a file-exists check and
    then fails deep inside petitRADTRANS. This drives the real transfer rather than a
    stand-in, because the staging and the clean-up are the behaviour in question and a
    stand-in could only demonstrate itself.
    """
    body = prt_data.HDF5_MAGIC + b'table contents'
    dest = (
        tree
        / 'opacities/lines/correlated_k/H2O/1H2-16O'
        / FAKE_FILES['opacities/lines/correlated_k/H2O/1H2-16O']
    )

    monkeypatch.setattr(prt_data.urllib.request, 'urlopen', _serve(_FakeResponse(body)))
    prt_data._download('/opacities/lines/correlated_k/H2O/1H2-16O/water.h5', dest)
    assert dest.read_bytes() == body
    assert list(tree.rglob('*.part')) == [], 'the staged file is cleared once it lands'

    # A transfer that dies after the first chunk: nothing survives, not even a stub.
    dest.unlink()
    monkeypatch.setattr(
        prt_data.urllib.request,
        'urlopen',
        _serve(_FakeResponse(body, fail_after=4)),
    )
    with pytest.raises(OSError, match='connection reset'):
        prt_data._download('/opacities/lines/correlated_k/H2O/1H2-16O/water.h5', dest)
    assert not dest.exists()
    assert list(tree.rglob('*.part')) == []
