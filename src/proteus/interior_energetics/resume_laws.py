"""Keep the separation laws of a resumed Aragog run.

``init_coupler.toml`` in the run directory records the config the run last started
with as loaded, with ``RECORD_OVERRIDES`` applied. ``Proteus.start()`` writes it again on every
start (a resume once every resume check has passed), so a resume compares the new
config with it first. A changed ``separation_viscosity`` or ``separation_buoyancy``
would switch the melt drainage law at the resume row; the resume stops unless
``interior_energetics.aragog.allow_resume_law_change`` is set.

The mesh end cells (``cmb_cell_thickness``, ``surface_cell_thickness``) must stay: the
snapshot entropy is put back node by node, so other end cells would move every node. The
allow switch does not cover them. A record without a cell field does not tell the mesh of
the run; ``aragog_resume.refuse_another_mesh`` then compares the radii of the snapshot. ``rheology.mlt_bottom_slope`` is a law like the two above
when the rheology is on in the run and in the resume.

``interior_energetics.mixing_length`` is a law of an Aragog and of a SPIDER run alike; a record of
either module or of no named module holds it, one of a module without convection does not.

A record that names the Aragog module and lacks ``separation_viscosity`` counts as
aragog's ``'melt'``, one with the rheology on and no slope as 1.0. A record of another interior module holds no Aragog law. Any law
the record does not hold must be set in the config file of the resume; the allow switch
does not stand in for a missing record.
"""

from __future__ import annotations

import logging
import os
import tomllib

log = logging.getLogger('fwl.' + __name__)

LAW_FIELDS = ('separation_viscosity', 'separation_buoyancy')
UNRECORDED_VISCOSITY = 'melt'  # aragog's default, which a record without the field ran under
RHEOLOGY_KEY = 'interior_energetics.aragog.rheology.enabled'
LID_KEY = 'interior_energetics.aragog.lid_mush_base'
SLOPE_KEY = 'interior_energetics.aragog.rheology.mlt_bottom_slope'
MIXING_KEY = 'interior_energetics.mixing_length'
MLT_MODULES = (
    'aragog',
    'spider',
)  # interior modules whose convective flux reads the mixing length
MESH_FIELDS = ('cmb_cell_thickness', 'surface_cell_thickness')
# The slope a record without the field ran under.
UNRECORDED_SLOPE = 1.0
# The permission to switch a separation law holds for one resume, so the record keeps it off.
RECORD_OVERRIDES = {'interior_energetics.aragog.allow_resume_law_change': False}


class ResumeSeparationLawError(RuntimeError):
    """A resume would change a separation law the run was written with."""


def _interior_table(path) -> dict:
    """Read ``path``; return its ``interior_energetics`` table ({} if absent or unreadable)."""
    try:
        with open(path, 'rb') as f:
            raw = tomllib.load(f)
    except (OSError, ValueError):
        return {}
    interior = raw.get('interior_energetics')
    return interior if isinstance(interior, dict) else {}


def _aragog_table(path) -> tuple[str | None, dict]:
    """Read ``path``; return its ``interior_energetics.module`` (None if absent, unreadable
    or not a string) and its ``interior_energetics.aragog`` table."""
    interior = _interior_table(path)
    aragog = interior.get('aragog')
    module = interior.get('module')
    return module if isinstance(module, str) else None, aragog if isinstance(
        aragog, dict
    ) else {}


def _where(stored_path: str, module, modules=('aragog',)) -> str:
    """The record named in a refusal, with why it holds no law when its module is not in ``modules``."""
    if module is None:
        return f'{stored_path}, which is missing, unreadable or names no interior module'
    if module not in modules:
        return f'{stored_path}, which names interior module {module!r}'
    return stored_path


def _unrecorded(key: str, stored_path: str, module, config_path) -> str:
    """Refusal text for a law the run did not record."""
    where = _where(stored_path, module)
    if module != 'aragog':
        hint = 'the law of the run is unknown'
    else:
        hint = (
            "a run without the field ran under 'solid' with an aragog below 26.10.22 and "
            "under 'mixture' from 26.10.22 (the 'aragog version' line of the run's "
            'proteus_NN.log files written before this attempt)'
        )
    return (
        f'{key} is not recorded in {where}: {hint}; set it in {config_path} to the law the '
        'run used (allow_resume_law_change does not cover a law the run did not record)'
    )


def _check_mesh(aragog, stored, stored_path, config_path, refusals, notes) -> None:
    """Refuse a resume on other mesh end cells than the run's; warn where they are unknown."""
    old = {f: stored[f] for f in MESH_FIELDS if f in stored}
    # with no CMB cell on either side, 0 and 'uniform' at the surface are both the uniform mesh
    no_cmb = old.get('cmb_cell_thickness') == aragog.cmb_cell_thickness == 0
    for field in MESH_FIELDS:
        new = getattr(aragog, field)
        if field not in old:
            notes.append(
                (
                    logging.WARNING,
                    f'interior_energetics.aragog.{field} is not recorded in {stored_path}; the '
                    'resume checks its mesh against the radii of the interior snapshot',
                )
            )
            continue
        if old[field] == new or (
            no_cmb
            and field == 'surface_cell_thickness'
            and old[field] in (0, 'uniform')
            and new in (0, 'uniform')
        ):
            continue
        refusals.append(
            f'interior_energetics.aragog.{field} is {new!r} in {config_path}, but the run was '
            f'written with {old[field]!r} ({stored_path}); the snapshot entropy is put back node by '
            f'node, so the resume must keep the mesh of the run: set it to {old[field]!r} '
            '(allow_resume_law_change does not cover the mesh)'
        )


def check_resume_separation_laws(
    config, config_path, output_dir
) -> list[tuple[str, object, str]]:
    """Compare the separation laws of a resume with those the run recorded.

    Parameters
    ----------
    config : proteus.config.Config
        Configuration of the resume.
    config_path : str or pathlib.Path
        Config file of the resume, read to see which laws it sets.
    output_dir : str or pathlib.Path
        Run directory holding the ``init_coupler.toml`` of the earlier start.

    Returns
    -------
    list of tuple
        ``(field, recorded value, new value)`` for each law the resume changes; empty
        when the laws stay, and when the resume uses neither Aragog nor SPIDER. A SPIDER
        resume compares only the mixing length.

    Raises
    ------
    ResumeSeparationLawError
        If a law changes and ``allow_resume_law_change`` is false, or the run did not
        record a law and the config file does not set it. Nothing is logged then.
    """
    stored_path = os.path.join(output_dir, 'init_coupler.toml')
    module, stored = _aragog_table(stored_path)
    if config.interior_energetics.module not in MLT_MODULES:
        return []
    if module not in (None, 'aragog'):  # a record of another module holds no Aragog law
        stored = {}
    aragog = config.interior_energetics.aragog
    in_file = _aragog_table(config_path)[1]
    switches, refusals, notes = [], [], []

    def switch(field, key, old, new, source=''):
        """Record a changed law: a switch with a warning when allowed, else a refusal."""
        switches.append((field, old, new))
        if aragog.allow_resume_law_change:
            notes.append(
                (
                    logging.WARNING,
                    f'Resume switches {key} from {old!r} to {new!r} (allow_resume_law_change = true)',
                )
            )
        else:
            refusals.append(
                f'{key} is {new!r} in {config_path}, but the run was written with {old!r}{source}; '
                f'set it to {old!r} to keep the law, or set interior_energetics.aragog.'
                'allow_resume_law_change = true to switch it at the resume'
            )

    new = config.interior_energetics.mixing_length
    recorded = _interior_table(stored_path) if module in (None, *MLT_MODULES) else {}
    if 'mixing_length' in recorded:
        if recorded['mixing_length'] != new:
            switch(
                'mixing_length', MIXING_KEY, recorded['mixing_length'], new, f' ({stored_path})'
            )
    elif 'mixing_length' in _interior_table(config_path):
        notes.append(
            (
                logging.INFO,
                f'{MIXING_KEY} is not recorded for this run; it resumes under {new!r}',
            )
        )
    else:
        refusals.append(
            f'{MIXING_KEY} is not recorded in {_where(stored_path, module, MLT_MODULES)}: the '
            f'mixing length of the run is unknown; set it in {config_path} to the value the run '
            'used (allow_resume_law_change does not cover a law the run did not record)'
        )
    if config.interior_energetics.module != 'aragog':
        return _finish(switches, refusals, notes)

    for field in LAW_FIELDS:
        key = f'interior_energetics.aragog.{field}'
        new = getattr(aragog, field)
        if field in stored:
            old, source = stored[field], stored_path
        elif field == 'separation_viscosity' and module == 'aragog':
            old, source = (
                UNRECORDED_VISCOSITY,
                f"not recorded in {stored_path}: aragog's default",
            )
        elif field in in_file:
            notes.append(
                (logging.INFO, f'{key} is not recorded for this run; it resumes under {new!r}')
            )
            continue
        else:
            refusals.append(_unrecorded(key, stored_path, module, config_path))
            continue
        if old != new:
            switch(field, key, old, new, f' ({source})')
    # A record without the rheology table ran with the rheology off, one without
    # lid_mush_base without the lid; with no Aragog table at all neither is known.
    if module is None and not stored:
        for key, new in (
            (RHEOLOGY_KEY, aragog.rheology.enabled),
            (LID_KEY, aragog.lid_mush_base),
        ):
            notes.append(
                (logging.INFO, f'{key} is not recorded for this run; it resumes under {new!r}')
            )
    else:
        rheology = stored.get('rheology')
        old = rheology.get('enabled', False) if isinstance(rheology, dict) else False
        if old != aragog.rheology.enabled:
            switch('rheology.enabled', RHEOLOGY_KEY, old, aragog.rheology.enabled)
        old = bool(stored.get('lid_mush_base', False))
        if old != aragog.lid_mush_base:
            switch('lid_mush_base', LID_KEY, old, aragog.lid_mush_base)
    _check_mesh(aragog, stored, stored_path, config_path, refusals, notes)
    rheology = stored.get('rheology')
    if aragog.rheology.enabled and isinstance(rheology, dict) and rheology.get('enabled'):
        old = rheology.get('mlt_bottom_slope', UNRECORDED_SLOPE)
        if old != aragog.rheology.mlt_bottom_slope:
            switch(
                'rheology.mlt_bottom_slope', SLOPE_KEY, old, aragog.rheology.mlt_bottom_slope
            )
    return _finish(switches, refusals, notes)


def _finish(switches, refusals, notes) -> list[tuple[str, object, str]]:
    """Raise the collected refusals, or log the notes and return the switches."""
    if refusals:
        raise ResumeSeparationLawError('Resume refused:\n  ' + '\n  '.join(refusals))
    for level, note in notes:
        log.log(level, note)
    return switches
