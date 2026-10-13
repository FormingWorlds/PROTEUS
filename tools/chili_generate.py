#!/usr/bin/env python3
"""Generate the CHILI intercomparison configs from the tutorial configs.

The tutorial configs are the base of the CHILI cases:
``input/tutorials/tutorial_earth.toml`` and
``input/tutorials/tutorial_venus.toml`` hold the Earth and Venus setups
of the protocol, except the redox state, which they set to IW+2, and the
end time of the Venus tutorial, which is 3 Myr. This script derives
every case config from them by applying only the deltas
listed in ``CASE_DELTAS`` below and the protocol's redox state, so the
generated cases and the tutorials cannot drift apart in any other
field; the grid specs sweep the ``GRID_AXES``. The standalone grid
configs under ``input/tutorials/chili_grid/`` are not generated and
hold the protocol values by hand. The unit tests in
``tests/tools/test_chili_generate.py`` enforce that contract:
each generated config must equal its tutorial base on every field
outside its delta list, the grid axes must match the standalone grid
configs under ``input/tutorials/chili_grid/``, and the committed files
under ``input/chili/intercomp/`` must equal a fresh regeneration.

Cases
-----
- ``earth.toml`` / ``venus.toml``: tutorial configs with the output
  path and the protocol's redox state set; ``venus.toml`` also keeps an
  end time of 5 Gyr.
- ``earth.grid.toml`` / ``venus.grid.toml``: ``proteus grid`` specs
  sweeping the protocol H and C inventories around the nominal cases.
- ``tr1b.toml`` / ``tr1e.toml`` / ``tr1a.toml``: TRAPPIST-1 b, e, and
  alpha (protocol Table 4), derived from the Earth base with the
  TRAPPIST star, orbit, and integration-time deltas and the protocol's
  redox state. Escape is disabled in these cases: ZEPHYRUS pairs only
  with the Spada evolution tracks, and a 0.09 M_sun star needs the
  Baraffe tracks, so the escape treatment for the TRAPPIST cases awaits
  a protocol decision. The configs are generated for completeness;
  their runs are validated separately.

Usage
-----
    python tools/chili_generate.py

Writes the configs to ``input/chili/intercomp/``.
"""

from __future__ import annotations

import os

import tomlkit

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TUTORIALS = os.path.join(ROOT, 'input', 'tutorials')
OUTDIR = os.path.join(ROOT, 'input', 'chili', 'intercomp')

# Per-case deltas applied on top of the tutorial base config. Keys are dotted config
# paths; every path listed here, and the redox state that ``case_deltas`` adds for all
# cases, is exempt from the equality contract with the tutorial base.
CASE_DELTAS: dict[str, dict] = {
    # Nominal cases: the tutorials at the protocol's redox state, written
    # to the intercomparison output folders.
    'earth': {
        'base': 'tutorial_earth.toml',
        'set': {
            'params.out.path': 'chili_earth',
        },
    },
    'venus': {
        'base': 'tutorial_venus.toml',
        'set': {
            'params.out.path': 'chili_venus',
            # The tutorial stops at 3 Myr; the value is written as in the tutorial files.
            'params.stop.time.maximum': tomlkit.value('5e9'),
        },
    },
    # TRAPPIST-1 cases (protocol Table 4). Flux scaling follows the
    # tutorial convention: the MORS track luminosity is used directly
    # (star.bol_scale stays at its default of 1).
    'tr1b': {
        'base': 'tutorial_earth.toml',
        'set': {
            'params.out.path': 'chili_tr1b',
            # ZEPHYRUS requires the Spada tracks; escape for the
            # TRAPPIST cases awaits a protocol decision.
            'escape.module': 'none',
            'params.stop.time.maximum': 7.55e9,  # to present day (7.6 Gyr)
            'star.mass': 0.0898,
            'star.age_ini': 0.05,
            'star.mors.tracks': 'baraffe',
            'star.mors.age_now': 7.6,
            'star.mors.star_name': 'trappist-1',
            'star.mors.spectrum_source': 'muscles',
            'orbit.semimajoraxis': 0.01154,
        },
    },
    'tr1e': {
        'base': 'tutorial_earth.toml',
        'set': {
            'params.out.path': 'chili_tr1e',
            # ZEPHYRUS requires the Spada tracks; escape for the
            # TRAPPIST cases awaits a protocol decision.
            'escape.module': 'none',
            'params.stop.time.maximum': 7.55e9,
            'star.mass': 0.0898,
            'star.age_ini': 0.05,
            'star.mors.tracks': 'baraffe',
            'star.mors.age_now': 7.6,
            'star.mors.star_name': 'trappist-1',
            'star.mors.spectrum_source': 'muscles',
            'orbit.semimajoraxis': 0.02925,
        },
    },
    'tr1a': {
        'base': 'tutorial_earth.toml',
        'set': {
            'params.out.path': 'chili_tr1a',
            # ZEPHYRUS requires the Spada tracks; escape for the
            # TRAPPIST cases awaits a protocol decision.
            'escape.module': 'none',
            'params.stop.time.maximum': 6.6e9,  # later start (1 Gyr)
            'star.mass': 0.0898,
            'star.age_ini': 1.00,
            'star.mors.tracks': 'baraffe',
            'star.mors.age_now': 7.6,
            'star.mors.star_name': 'trappist-1',
            'star.mors.spectrum_source': 'muscles',
            'orbit.semimajoraxis': 6.750e-4,
        },
    },
}

# Every case runs at the oxygen fugacity of the CHILI protocol, relative to iron-wustite
# [log10 units], whatever the tutorial base sets.
PROTOCOL_FO2_SHIFT_IW = 4.0

# Protocol H and C inventory axes [kg] for the grid sweeps. The
# standalone per-case grid configs under input/tutorials/chili_grid/
# carry exactly these values; the lockstep test compares the two.
GRID_AXES: dict[str, list[float]] = {
    'planet.elements.H_budget': [1.60e20, 7.80e20, 1.60e21],
    'planet.elements.C_budget': [1.36e20, 2.73e20, 5.44e20],
}

# Grid-runner settings shared by both grid specs. max_mem is sized for
# a coupled AGNI + Aragog + Zalmoxis run, which peaks near 4 GB.
GRID_RUNNER = {
    'symlink': '',
    'use_slurm': False,
    'max_jobs': 9,
    'max_days': 1,
    'max_mem': 6,
}


def _set_dotted(doc, dotted: str, value) -> None:
    """Set ``dotted`` path in a tomlkit document to ``value``."""
    keys = dotted.split('.')
    node = doc
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value


def case_deltas(name: str) -> dict:
    """Deltas of case ``name``: its ``CASE_DELTAS`` entries and the protocol's redox state.

    Raises
    ------
    ValueError
        If the case lists a redox state of its own.
    """
    deltas = CASE_DELTAS[name]['set']
    if 'outgas.fO2_shift_IW' in deltas:
        raise ValueError(f'{name}: PROTOCOL_FO2_SHIFT_IW sets the redox state of all cases')
    return {**deltas, 'outgas.fO2_shift_IW': PROTOCOL_FO2_SHIFT_IW}


def generate_case(name: str) -> str:
    """Render one intercomparison case from its tutorial base.

    Returns the TOML text. tomlkit round-trips the tutorial file, so
    everything outside the delta list stays byte-identical to the base,
    apart from the leading comment block, which is replaced with a
    case-specific header (the tutorial header describes the tutorial,
    not the derived case).
    """
    spec = CASE_DELTAS[name]
    base_path = os.path.join(TUTORIALS, spec['base'])
    with open(base_path) as f:
        text = f.read()
    doc = tomlkit.parse(text)
    for dotted, value in case_deltas(name).items():
        _set_dotted(doc, dotted, value)
    rendered = tomlkit.dumps(doc)

    # Replace the tutorial's leading comment block (everything up to
    # the first non-comment, non-blank line) with the case header.
    lines = rendered.split('\n')
    body_start = None
    for i, line in enumerate(lines):
        if line.strip() and not line.lstrip().startswith('#'):
            body_start = i
            break
    if body_start is None:
        # Comment-only or empty document: nothing to keep below the header.
        body_start = len(lines)
    header = [
        f'# CHILI intercomparison case: {name}',
        f'# Generated by tools/chili_generate.py from {spec["base"]};',
        '# do not edit by hand (see input/chili/intercomp/README.md).',
        '',
    ]
    return '\n'.join(header + lines[body_start:])


def generate_grid(planet: str) -> str:
    """Render the ``proteus grid`` spec sweeping H and C for ``planet``."""
    doc = tomlkit.document()
    doc['output'] = f'chili_{planet}_grid'
    doc['ref_config'] = f'input/chili/intercomp/{planet}.toml'
    for key, value in GRID_RUNNER.items():
        doc[key] = value
    for dotted, values in GRID_AXES.items():
        table = tomlkit.table()
        table['method'] = 'direct'
        table['values'] = values
        doc[tomlkit.key(dotted)] = table
    return tomlkit.dumps(doc)


def main() -> None:
    """Write every case and grid spec; a case that cannot be rendered leaves all files as they are."""
    texts = {f'{name}.toml': generate_case(name) for name in CASE_DELTAS}
    texts |= {f'{planet}.grid.toml': generate_grid(planet) for planet in ('earth', 'venus')}
    os.makedirs(OUTDIR, exist_ok=True)
    for fname, text in texts.items():
        path = os.path.join(OUTDIR, fname)
        with open(path, 'w') as f:
            f.write(text)
        print(f'wrote {os.path.relpath(path, ROOT)}')


if __name__ == '__main__':
    main()
