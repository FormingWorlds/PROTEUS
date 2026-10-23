"""Keep the separation laws of a resumed Aragog run.

``init_coupler.toml`` in the run directory records the config the run last started
with. ``Proteus.start()`` writes it again on every start, so a resume compares the
new config with it before that write. A changed ``separation_viscosity`` or
``separation_buoyancy`` would switch the melt drainage law at the resume row; the
resume stops unless ``interior_energetics.aragog.allow_resume_law_change`` is set.

A run without a recorded ``separation_viscosity`` is older than the field and ran
under aragog's ``'melt'``. A run without a recorded ``separation_buoyancy`` is older
than that field and ran under ``'solid'`` with an aragog below 26.10.22 and under
``'mixture'`` from 26.10.22, so it resumes only when the config file sets the field.
"""

from __future__ import annotations

import logging
import os
import tomllib

log = logging.getLogger('fwl.' + __name__)

LAW_FIELDS = ('separation_viscosity', 'separation_buoyancy')
UNRECORDED_VISCOSITY = 'melt'  # aragog's default before PROTEUS passed the field


class ResumeSeparationLawError(RuntimeError):
    """A resume would change a separation law the run was written with."""


def _aragog_table(path) -> tuple[bool, dict, dict]:
    """Read ``path``; return (readable, interior_energetics table, its aragog table)."""
    try:
        with open(path, 'rb') as f:
            raw = tomllib.load(f)
    except (OSError, ValueError):
        return False, {}, {}
    interior = raw.get('interior_energetics')
    interior = interior if isinstance(interior, dict) else {}
    aragog = interior.get('aragog')
    return True, interior, aragog if isinstance(aragog, dict) else {}


def check_resume_separation_laws(
    config, config_path, output_dir
) -> list[tuple[str, object, str]]:
    """Compare the separation laws of a resume with those the run recorded.

    Parameters
    ----------
    config : proteus.config.Config
        Configuration of the resume.
    config_path : str or pathlib.Path
        Config file of the resume, read to see whether it sets ``separation_buoyancy``.
    output_dir : str or pathlib.Path
        Run directory holding the ``init_coupler.toml`` of the earlier start.

    Returns
    -------
    list of tuple
        ``(field, recorded value, new value)`` for each law the resume changes; empty
        when the laws stay, and when the run or the resume does not use Aragog.

    Raises
    ------
    ResumeSeparationLawError
        If a law changes and ``allow_resume_law_change`` is false, or the run recorded
        no ``separation_buoyancy`` and the config file does not set it.
    """
    stored_path = os.path.join(output_dir, 'init_coupler.toml')
    readable, interior, stored = _aragog_table(stored_path)
    if (
        config.interior_energetics.module != 'aragog'
        or interior.get('module', 'aragog') != 'aragog'
    ):
        return []
    aragog = config.interior_energetics.aragog
    switches, refusals, sources = [], [], {}
    for field in LAW_FIELDS:
        key = f'interior_energetics.aragog.{field}'
        new = getattr(aragog, field)
        if field in stored:
            old, sources[field] = stored[field], stored_path
        elif field == 'separation_viscosity':
            old, sources[field] = (
                UNRECORDED_VISCOSITY,
                f"not recorded in {stored_path}: aragog's default",
            )
        elif field in _aragog_table(config_path)[2]:
            log.info('%s is not recorded for this run; it resumes under %r', key, new)
            continue
        else:
            refusals.append(
                f'{key} is not recorded in {stored_path}'
                + ('' if readable else ' (missing or unreadable)')
                + ": a run written before the field ran under 'solid' with an aragog below "
                "26.10.22 and under 'mixture' from 26.10.22 (the 'aragog version' line of the "
                f"run's proteus_NN.log); set it in {config_path} to the law the run used"
            )
            continue
        if old != new:
            switches.append((field, old, new))
    if switches and aragog.allow_resume_law_change:
        for field, old, new in switches:
            log.warning(
                'Resume switches interior_energetics.aragog.%s from %r to %r '
                '(allow_resume_law_change = true)',
                field,
                old,
                new,
            )
    elif switches:
        refusals = [
            f'interior_energetics.aragog.{field} is {new!r} in {config_path}, but the run '
            f'was written with {old!r} ({sources[field]}); set it '
            f'to {old!r} to keep the law, or set interior_energetics.aragog.'
            'allow_resume_law_change = true to switch it at the resume'
            for field, old, new in switches
        ] + refusals
    if refusals:
        raise ResumeSeparationLawError('Resume refused:\n  ' + '\n  '.join(refusals))
    return switches
