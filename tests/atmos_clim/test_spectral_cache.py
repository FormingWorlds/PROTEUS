"""
Unit tests for reuse of prepared spectral files across runs.

Covers `proteus.atmos_clim.spectral_cache`: what makes two runs share a
prepared file, that the pair is seeded and stored together, and that a cache
which cannot be read or written costs a run time rather than correctness.

References:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import pytest

from proteus.atmos_clim.spectral_cache import (
    SPECTRAL_SUFFIXES,
    cache_key,
    seed_from_cache,
    store_in_cache,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _make_inputs(tmp_path, star_bytes=b'1.0 2.0\n3.0 4.0\n'):
    """Write a base spectral file and a stellar spectrum, and return both."""
    base = tmp_path / 'Honeyside.sf'
    base.write_bytes(b'x' * 4096)
    star = tmp_path / '0.sflux'
    star.write_bytes(star_bytes)
    return base, star


def _make_prepared(out_dir, marker=b'prepared'):
    """Write the prepared pair a finished build leaves in an output folder."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in SPECTRAL_SUFFIXES:
        (out_dir / f'runtime.sf{suffix}').write_bytes(marker + suffix.encode())


@pytest.mark.unit
def test_key_tracks_the_stellar_spectrum_and_the_spectral_resolution(tmp_path):
    """Two runs share a prepared file only when every input that goes into
    building it agrees. The stellar spectrum is hashed by content, so a sweep
    over any star parameter changes the key without this module having to know
    which config fields those are.
    """
    base, star = _make_inputs(tmp_path)
    key = cache_key(base, star, 'Honeyside', '48')

    # Same inputs, same entry: this is what lets a study reuse one build.
    assert cache_key(base, star, 'Honeyside', '48') == key

    # A different stellar spectrum is a different file, even byte-for-byte the
    # same length, so a content hash rather than a size check is required.
    other_star = tmp_path / 'other.sflux'
    other_star.write_bytes(b'9.0 2.0\n3.0 4.0\n')
    assert other_star.stat().st_size == star.stat().st_size
    assert cache_key(base, other_star, 'Honeyside', '48') != key

    # Resolution and group select a different base file, so neither may collide.
    assert cache_key(base, star, 'Honeyside', '256') != key
    assert cache_key(base, star, 'Frostflow', '48') != key


@pytest.mark.unit
def test_key_changes_when_the_base_spectral_file_is_updated(tmp_path):
    """A FWL_DATA update must not be served a stale entry. The base file is
    fingerprinted by size and modification time rather than hashed, because it
    is large and read-only, so both are exercised here.
    """
    base, star = _make_inputs(tmp_path)
    key = cache_key(base, star, 'Honeyside', '48')

    # Same size, newer file: the mtime component has to carry this one.
    stat = base.stat()
    base.write_bytes(b'y' * 4096)
    import os

    os.utime(base, (stat.st_atime, stat.st_mtime + 120))
    assert base.stat().st_size == 4096
    key_touched = cache_key(base, star, 'Honeyside', '48')
    assert key_touched != key

    # Same mtime, different size: the size component has to carry this one.
    base.write_bytes(b'y' * 8192)
    os.utime(base, (stat.st_atime, stat.st_mtime + 120))
    assert cache_key(base, star, 'Honeyside', '48') != key_touched


@pytest.mark.unit
def test_a_stored_entry_is_seeded_back_for_a_later_run(tmp_path):
    """The round trip a study relies on: the first run stores what it built and
    every later run with the same inputs starts from it. Both files of the pair
    travel together, because the module that consumes them checks only the
    first and would otherwise run with a missing companion.
    """
    cache = tmp_path / 'cache'
    first = tmp_path / 'run_0'
    _make_prepared(first)

    assert store_in_cache(cache, 'abc123', first) is True

    second = tmp_path / 'run_1'
    assert seed_from_cache(cache, 'abc123', second) is True
    for suffix in SPECTRAL_SUFFIXES:
        seeded = second / f'runtime.sf{suffix}'
        assert seeded.is_file()
        assert seeded.read_bytes() == (first / f'runtime.sf{suffix}').read_bytes()

    # Discrimination: a key nothing was stored under is a miss, so the run
    # builds its own file rather than silently reusing another star's.
    third = tmp_path / 'run_2'
    assert seed_from_cache(cache, 'def456', third) is False
    assert not (third / 'runtime.sf').exists()


@pytest.mark.unit
def test_a_half_written_entry_is_not_served(tmp_path):
    """An entry is usable only once both files are in place. A reader that
    accepted the first alone would hand a run a prepared file whose companion
    is missing, which is worse than a miss because the run would not rebuild.
    """
    cache = tmp_path / 'cache'
    cache.mkdir()
    # Only the first of the pair, as a reader would see mid-store.
    (cache / 'abc123.sf').write_bytes(b'prepared')

    out = tmp_path / 'run'
    assert seed_from_cache(cache, 'abc123', out) is False
    assert not (out / 'runtime.sf').exists()

    # Discrimination: completing the pair makes the same key usable, so the
    # refusal above came from the missing companion and not from the reader
    # rejecting every entry.
    (cache / 'abc123.sf_k').write_bytes(b'prepared_k')
    assert seed_from_cache(cache, 'abc123', out) is True
    assert (out / 'runtime.sf').is_file()


@pytest.mark.unit
def test_a_run_that_built_nothing_stores_nothing(tmp_path):
    """Storing is driven by what the run produced, not by being asked. A run
    whose build failed before writing the pair must not publish a partial entry
    that every later run with the same star would then be served.
    """
    cache = tmp_path / 'cache'
    empty = tmp_path / 'run_empty'
    empty.mkdir()

    assert store_in_cache(cache, 'abc123', empty) is False
    assert not (cache / 'abc123.sf').exists()

    # Edge case: one file of the pair present is still nothing worth storing.
    (empty / 'runtime.sf').write_bytes(b'prepared')
    assert store_in_cache(cache, 'abc123', empty) is False
    assert not (cache / 'abc123.sf').exists()

    # Discrimination: the complete pair does store, so the refusals above came
    # from the missing file and not from a writer that always declines.
    (empty / 'runtime.sf_k').write_bytes(b'prepared_k')
    assert store_in_cache(cache, 'abc123', empty) is True
    assert (cache / 'abc123.sf').is_file()


@pytest.mark.unit
def test_an_unusable_cache_costs_time_and_not_correctness(tmp_path):
    """A cache that cannot be written or read leaves the run to build its own
    file. Bookkeeping around a simulation must not decide whether it runs, so
    both directions report the failure and return rather than raising.
    """
    # A file where the cache folder should be: it cannot be created beneath.
    blocked = tmp_path / 'not_a_directory'
    blocked.write_text('this is a file, so no folder can be made beneath it')

    run = tmp_path / 'run'
    _make_prepared(run)
    assert store_in_cache(blocked, 'abc123', run) is False

    # Reading a cache folder that was never created is a miss, not an error.
    assert seed_from_cache(tmp_path / 'never_made', 'abc123', tmp_path / 'out') is False

    # Discrimination: against a usable folder the same call succeeds, so the
    # False above came from the blocked path rather than from a writer that
    # always fails.
    assert store_in_cache(tmp_path / 'cache', 'abc123', run) is True


@pytest.mark.unit
def test_storing_twice_leaves_one_usable_entry(tmp_path):
    """Workers race to populate the same key, because they start together and
    share a stellar spectrum. The later store replaces the earlier one in
    place, so a reader sees one entry or the other and never a mixture.
    """
    cache = tmp_path / 'cache'
    first = tmp_path / 'run_0'
    second = tmp_path / 'run_1'
    _make_prepared(first, marker=b'from_first')
    _make_prepared(second, marker=b'from_second')

    assert store_in_cache(cache, 'abc123', first) is True
    assert store_in_cache(cache, 'abc123', second) is True

    # No temporary files survive to be mistaken for entries.
    assert sorted(p.name for p in cache.iterdir()) == ['abc123.sf', 'abc123.sf_k']

    out = tmp_path / 'run_2'
    assert seed_from_cache(cache, 'abc123', out) is True
    assert (out / 'runtime.sf').read_bytes() == b'from_second'
    assert (out / 'runtime.sf_k').read_bytes() == b'from_second_k'
