"""Lockstep tests for the CHILI intercomparison generator.

The tutorial configs are the base of the CHILI cases
(tools/chili_generate.py). These tests pin that contract:

- each generated case equals its tutorial base on every config field
  outside its declared delta list, via the real config loader;
- the grid-spec inventory axes match the standalone grid configs under
  input/tutorials/chili_grid/;
- every CHILI config carries the default stop block plus PROTOCOL_STOP,
  and PROTOCOL_STOP holds the protocol values;
- the committed files under input/chili/intercomp/ equal a fresh
  regeneration, so a tutorial retune without regeneration fails CI.

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import importlib.util
import logging
import re
from pathlib import Path

import attrs
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

REPO = Path(__file__).resolve().parents[2]


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        'chili_generate', REPO / 'tools' / 'chili_generate.py'
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_config(path):
    logging.disable(logging.WARNING)
    try:
        from proteus.config import read_config_object

        return read_config_object(path)
    finally:
        logging.disable(logging.NOTSET)


def _diff_paths(a, b, path=''):
    """Dotted paths of every leaf field that differs between two configs."""
    out = []
    if attrs.has(type(a)) and attrs.has(type(b)) and type(a) is type(b):
        for f in attrs.fields(type(a)):
            va, vb = getattr(a, f.name), getattr(b, f.name)
            out += _diff_paths(va, vb, f'{path}.{f.name}' if path else f.name)
    elif a != b:
        out.append(path)
    return out


def test_generated_cases_match_tutorials_outside_their_deltas():
    """Every case equals its tutorial base except the declared deltas.

    The contract that keeps the intercomparison and the tutorials in
    lockstep: a physics field changed in only one of the two surfaces
    appears here as an undeclared difference. The nominal cases also pin
    the boundary from the other side: they differ from their tutorial in
    the output path, the redox state and the stop rule only, and the redox
    state differs in every case.
    """
    gen = _load_generator()
    for name, spec in gen.CASE_DELTAS.items():
        tut = _read_config(str(REPO / 'input' / 'tutorials' / spec['base']))
        case = _read_config(str(REPO / 'input' / 'chili' / 'intercomp' / f'{name}.toml'))
        diffs = set(_diff_paths(tut, case))
        assert 'outgas.fO2_shift_IW' in diffs, name
        declared = gen.case_deltas(name)
        undeclared = diffs - set(declared)
        assert not undeclared, f'{name}: differs from tutorial outside its deltas: {undeclared}'
        # Every declared delta must land in the loaded config with its
        # declared value: a generator regression that dropped a delta
        # would leave the tutorial value in place. Deltas that coincide
        # with the base value (a documented pin, e.g. tr1b age_ini) are
        # covered by the same direct comparison.
        for dotted, want in declared.items():
            got = case
            for part in dotted.split('.'):
                got = getattr(got, part)
            if want == 'none':
                # The config loader maps the TOML 'none' sentinel to None.
                assert got is None, f'{name}: {dotted}={got!r}, want the none sentinel'
            elif isinstance(want, float):
                assert got == pytest.approx(want, rel=1e-12), f'{name}: {dotted}={got!r}'
            else:
                assert got == want, f'{name}: {dotted}={got!r}, want {want!r}'
    # Discrimination: the nominal cases carry these deltas and no other.
    nominal = {
        'params.out.path',
        'outgas.fO2_shift_IW',
        *gen.PROTOCOL_STOP,
        *gen.RUN_SETTINGS,
    }
    assert set(gen.case_deltas('earth')) == nominal
    assert set(gen.case_deltas('venus')) == nominal
    assert set(gen.CASE_DELTAS) == {'earth', 'venus', 'tr1b', 'tr1e', 'tr1a'}


def test_a_case_cannot_set_its_own_redox_state():
    """The redox state of the protocol is one value for all cases."""
    gen = _load_generator()
    assert 'outgas.fO2_shift_IW' not in gen.CASE_DELTAS['tr1a']['set']
    assert gen.case_deltas('tr1a')['outgas.fO2_shift_IW'] == pytest.approx(4.0)
    gen.CASE_DELTAS['tr1a']['set']['outgas.fO2_shift_IW'] = 3.0
    with pytest.raises(ValueError, match='tr1a: PROTOCOL_FO2_SHIFT_IW'):
        gen.generate_case('tr1a')


@pytest.mark.parametrize('key', _load_generator().PROTOCOL_STOP)
def test_a_case_cannot_set_its_own_stop_rule(key):
    """The stop rule of the protocol is one rule for all cases."""
    gen = _load_generator()
    assert key not in gen.CASE_DELTAS['venus']['set']
    assert gen.case_deltas('venus')[key] == gen.PROTOCOL_STOP[key]
    gen.CASE_DELTAS['venus']['set'][key] = 1
    with pytest.raises(ValueError, match=rf"venus: .*\['{re.escape(key)}'\]"):
        gen.generate_case('venus')


def test_every_chili_config_stops_only_by_the_protocol_rule():
    """The 5 cases and the 9 grid configs carry the default stop block plus PROTOCOL_STOP.

    The whole ``params.stop`` is compared, so a criterion turned on or off
    outside the protocol keys (solid.enabled, strict, stall) fails as well.
    """
    from proteus.config._params import StopParams

    gen = _load_generator()
    assert gen.PROTOCOL_STOP == {
        'params.stop.solid.phi_crit': pytest.approx(0.05, rel=1e-12),
        'params.stop.time.maximum': pytest.approx(1.0e9, rel=1e-12),
        'params.stop.radeqm.enabled': False,
        'params.stop.escape.enabled': False,
        'params.stop.iters.maximum': 1000000,
        'params.stop.clock.enabled': False,
    }
    want = attrs.asdict(StopParams())
    assert want['stall'] == {'enabled': True, 'maximum': 150}
    assert want['solid']['enabled'] is True
    assert want['time']['enabled'] is True
    assert want['strict'] is False
    for dotted, value in gen.PROTOCOL_STOP.items():
        table, key = dotted.split('.')[2:]
        want[table][key] = value
    files = [REPO / 'input' / 'chili' / 'intercomp' / f'{n}.toml' for n in gen.CASE_DELTAS]
    files += sorted((REPO / 'input' / 'tutorials' / 'chili_grid').glob('earth_H*_C*.toml'))
    assert len(files) == 14
    for path in files:
        assert attrs.asdict(_read_config(str(path)).params.stop) == want, path.name


def test_every_chili_config_takes_the_buoyancy_against_the_solid():
    """The 5 cases and the 9 grid configs set separation_buoyancy 'solid', the buoyancy form of the
    submitted runs; the PROTEUS default 'mixture' changes their mush numbers. A case cannot set its
    own value, and the generated key sits at the indent of its table."""
    gen = _load_generator()
    assert gen.RUN_SETTINGS == {'interior_energetics.aragog.separation_buoyancy': 'solid'}
    files = [REPO / 'input' / 'chili' / 'intercomp' / f'{n}.toml' for n in gen.CASE_DELTAS]
    files += sorted((REPO / 'input' / 'tutorials' / 'chili_grid').glob('earth_H*_C*.toml'))
    assert len(files) == 14
    for path in files:
        assert (
            _read_config(str(path)).interior_energetics.aragog.separation_buoyancy == 'solid'
        ), path.name
    for name in gen.CASE_DELTAS:
        text = (REPO / 'input' / 'chili' / 'intercomp' / f'{name}.toml').read_text()
        assert '\n        separation_buoyancy = "solid"\n' in text, name
    gen.CASE_DELTAS['tr1e']['set']['interior_energetics.aragog.separation_buoyancy'] = 'mixture'
    with pytest.raises(ValueError, match=r'tr1e: .*separation_buoyancy'):
        gen.generate_case('tr1e')


def test_protocol_values_are_fresh_items_in_each_case(tmp_path, monkeypatch):
    """A comment on a replaced tutorial line does not travel into the next case."""
    for name in ('tutorial_earth.toml', 'tutorial_venus.toml'):
        (tmp_path / name).write_text((REPO / 'input' / 'tutorials' / name).read_text())
    venus = tmp_path / 'tutorial_venus.toml'
    venus.write_text(venus.read_text().replace('maximum = 3e6', 'maximum = 3e6  # venus note'))
    gen = _load_generator()
    monkeypatch.setattr(gen, 'TUTORIALS', str(tmp_path))
    assert '# venus note' in venus.read_text()
    gen.generate_case('venus')
    assert '# venus note' not in gen.generate_case('tr1b')


@pytest.mark.parametrize('failure', ['case', 'grid'])
def test_a_refused_case_leaves_every_generated_file_as_it_was(tmp_path, failure):
    """A case or a grid spec that cannot be rendered stops ``main`` before any write."""
    gen = _load_generator()
    gen.OUTDIR = str(tmp_path)
    gen.main()
    files = sorted(tmp_path.iterdir())
    assert len(files) == 7
    for p in files:
        p.write_text('stale')
    if failure == 'case':
        gen.CASE_DELTAS['tr1a']['set']['outgas.fO2_shift_IW'] = 3.0
    else:
        gen.generate_grid = lambda planet: (_ for _ in ()).throw(ValueError('venus grid'))
    with pytest.raises(ValueError, match='tr1a' if failure == 'case' else 'venus grid'):
        gen.main()
    assert [p.read_text() for p in sorted(tmp_path.iterdir())] == ['stale'] * 7


def test_grid_axes_match_standalone_grid_configs():
    """The grid-spec inventory axes equal the standalone grid configs.

    The 3x3 standalone configs under input/tutorials/chili_grid/ carry
    exactly the protocol H and C budgets that the grid specs sweep; a
    change to either surface without the other breaks the pairing.
    """
    import tomllib

    gen = _load_generator()
    h_axis = sorted(gen.GRID_AXES['planet.elements.H_budget'])
    c_axis = sorted(gen.GRID_AXES['planet.elements.C_budget'])

    grid_dir = REPO / 'input' / 'tutorials' / 'chili_grid'
    files = sorted(grid_dir.glob('earth_H*_C*.toml'))
    # The full cross product must be present: 3 x 3 cases.
    assert len(files) == len(h_axis) * len(c_axis)

    budgets = set()
    for f in files:
        with open(f, 'rb') as hdl:
            cfg = tomllib.load(hdl)
        assert cfg['outgas']['fO2_shift_IW'] == pytest.approx(gen.PROTOCOL_FO2_SHIFT_IW), f.name
        ele = cfg['planet']['elements']
        budgets.add((float(ele['H_budget']), float(ele['C_budget'])))
    expected = {(h, c) for h in h_axis for c in c_axis}
    assert budgets == expected, (
        f'grid configs do not span the protocol axes: {budgets ^ expected}'
    )


def test_committed_intercomp_files_match_regeneration():
    """The committed intercomparison files equal a fresh regeneration.

    Retuning a tutorial without rerunning tools/chili_generate.py
    leaves stale committed configs; this comparison fails CI until the
    regeneration is committed. The comparison is on the parsed TOML
    content, so a serializer formatting change alone cannot fail it.
    """
    import tomllib

    gen = _load_generator()
    outdir = REPO / 'input' / 'chili' / 'intercomp'
    stale = []
    for name in gen.CASE_DELTAS:
        committed = tomllib.loads((outdir / f'{name}.toml').read_text())
        if committed != tomllib.loads(gen.generate_case(name)):
            stale.append(f'{name}.toml')
    for planet in ('earth', 'venus'):
        committed = tomllib.loads((outdir / f'{planet}.grid.toml').read_text())
        if committed != tomllib.loads(gen.generate_grid(planet)):
            stale.append(f'{planet}.grid.toml')
    assert not stale, f'stale committed configs, rerun tools/chili_generate.py: {stale}'
    # Discrimination: the comparison must be able to fail on a physics
    # field outside the delta list, not just on the output path.
    perturbed = gen.generate_case('earth').replace(
        'semimajoraxis = 1.0', 'semimajoraxis = 1.05'
    )
    assert perturbed != gen.generate_case('earth')  # the perturbation landed
    assert tomllib.loads(perturbed) != tomllib.loads((outdir / 'earth.toml').read_text())


def _stop_block(text):
    """Lines from the first [params.stop.*] header to the [star] header, inclusive."""
    lines = text.split('\n')
    start = next(i for i, ln in enumerate(lines) if ln.strip().startswith('[params.stop.'))
    end = lines.index('[star]')
    return lines[start : end + 1]


def test_committed_cases_are_the_generator_output_byte_for_byte():
    """Each committed case file is exactly the generator output, layout included."""
    gen = _load_generator()
    for name in gen.CASE_DELTAS:
        path = REPO / 'input' / 'chili' / 'intercomp' / f'{name}.toml'
        assert path.read_text() == gen.generate_case(name), name


def test_generated_stop_block_has_the_grid_layout():
    """Each case lays out its stop tables as the grid configs do, so both read alike."""
    gen = _load_generator()
    grid = (REPO / 'input' / 'tutorials' / 'chili_grid' / 'earth_Hlow_Clow.toml').read_text()
    for name in gen.CASE_DELTAS:
        assert _stop_block(gen.generate_case(name)) == _stop_block(grid), name
    assert _stop_block(grid)[-2:] == ['', '[star]']


def test_stop_block_layout_is_stable():
    """Laying out an already laid-out case changes nothing."""
    gen = _load_generator()
    text = gen.generate_case('earth')
    assert gen._format_stop_block(text, 'earth') == text


@pytest.mark.parametrize(
    ('text', 'match'),
    [
        ('[a]\n    x = 1\n', 'no \\[params.stop'),
        (
            '[params.stop.time]\nm = 1\n[params.dt]\nx = 1\n[params.stop.solid]\np = 0.1\n',
            'not in one block',
        ),
        (
            '[params.stop.time]\nv = [\n  1,\n]\n[params.stop.iters]\nm = 2\n[star]\n',
            "multi-line value.*'v = \\['",
        ),
        ('[params.stop.time]\nnote = """\n  a\n"""\n[star]\n', 'multi-line value'),
        (
            'd = """\n[params.stop.time]"""\n[params.stop.time]\nm = 1\n[star]\ny = 1\n',
            'changes a parsed value',
        ),
    ],
    ids=[
        'no_stop_table',
        'split_block',
        'array',
        'string',
        'stop_header_inside_a_string',
    ],
)
def test_stop_block_layout_refuses_what_it_cannot_place(text, match):
    """A missing stop table, a split block, a multi-line value or a layout that changes a
    value stops the generator and names the case; a stop header inside a string before the
    first stop table reaches the value check."""
    with pytest.raises(ValueError, match='case_x: .*' + match):
        _load_generator()._format_stop_block(text, 'case_x')


def test_stop_block_layout_keeps_comments_and_values():
    """Inline comments, a comment ending in '{', aligned and dashed keys, a comment line and
    a nan value keep their values; a comment that ends the block stays outside it."""
    import tomllib

    gen = _load_generator()
    text = (
        '[a]\nx = nan\n[params.stop.time]  # t\nmaximum  = 1e9  # cap {\n[params.stop.iters]\n'
        '# note\nmax-x = [1, 2]\n\n# star\n[star]\ny = 2'
    )
    out = gen._format_stop_block(text, 'case_x')
    assert out == (
        '[a]\nx = nan\n    [params.stop.time]  # t\n        maximum  = 1e9  # cap {\n'
        '    [params.stop.iters]\n        # note\n        max-x = [1, 2]\n\n# star\n[star]\ny = 2'
    )
    assert repr(tomllib.loads(out)) == repr(tomllib.loads(text))
    assert tomllib.loads(out)['params']['stop'] == {
        'time': {'maximum': 1e9},
        'iters': {'max-x': [1, 2]},
    }
    assert gen._format_stop_block('[a]\n[params.stop.time]\nmaximum = 1', 'case_x') == (
        '[a]\n    [params.stop.time]\n        maximum = 1\n'
    )


def test_stop_block_is_only_the_stop_child_tables():
    """A [params.stop] parent table before the block and a [params.stopwatch] table after it
    keep their place and indent; only the [params.stop.*] tables are laid out."""
    gen = _load_generator()
    text = '[params.stop]\nx = 1\n[params.stop.time]\nm = 1\n[params.stopwatch]\ny = 2\n[star]\nz = 3'
    assert gen._format_stop_block(text, 'case_x') == (
        '[params.stop]\nx = 1\n    [params.stop.time]\n        m = 1\n\n[params.stopwatch]\ny = 2\n[star]\nz = 3'
    )


def test_a_base_that_is_not_valid_toml_names_the_case(tmp_path, monkeypatch):
    """A tutorial base that tomlkit cannot parse stops the case with its name and base file."""
    for name in ('tutorial_earth.toml', 'tutorial_venus.toml'):
        (tmp_path / name).write_text((REPO / 'input' / 'tutorials' / name).read_text())
    venus = tmp_path / 'tutorial_venus.toml'
    venus.write_text(venus.read_text() + '\n[broken\n')
    gen = _load_generator()
    monkeypatch.setattr(gen, 'TUTORIALS', str(tmp_path))
    with pytest.raises(ValueError, match='venus: tutorial_venus.toml is not valid TOML'):
        gen.generate_case('venus')
    assert gen.generate_case('earth').startswith('# CHILI intercomparison case: earth')


def test_readme_and_page_name_the_stall_and_iteration_limits_of_the_configs():
    """README and page give the stall stop (an error after 150 iterations) and the iteration
    limit (1000000) that the 14 configs load; the configs take the stall stop from its default."""
    readme = (REPO / 'input' / 'chili' / 'intercomp' / 'README.md').read_text()
    page = (REPO / 'docs' / 'Tutorials' / 'chili_intercomparison.md').read_text()
    for name, text in (('README', readme), ('page', page)):
        flat = ' '.join(text.split())
        assert 'as an error after 150 consecutive iterations' in flat, name
    assert 'iteration limit of 1000000' in ' '.join(readme.split())
    assert 'a limit of 1,000,000 iterations (`params.stop.iters.maximum`)' in ' '.join(
        page.split()
    )
    paths = sorted((REPO / 'input' / 'chili' / 'intercomp').glob('*.toml'))
    paths += sorted((REPO / 'input' / 'tutorials' / 'chili_grid').glob('*.toml'))
    cfgs = [_read_config(str(f)) for f in paths if not f.name.endswith('.grid.toml')]
    assert len(cfgs) == 14
    loaded = {
        (c.params.stop.stall.enabled, c.params.stop.stall.maximum, c.params.stop.iters.maximum)
        for c in cfgs
    }
    assert loaded == {(True, 150, 1000000)}
