"""Keep the separation laws of a resumed Aragog run.

``init_coupler.toml`` in the run directory records the config the run last started
with, with ``RECORD_OVERRIDES`` applied. ``Proteus.start()`` writes it again on every
start (a resume once every resume check has passed), so a resume compares the new
config with it first. A changed ``separation_viscosity`` or ``separation_buoyancy``
would switch the melt drainage law at the resume row; the resume stops unless
``interior_energetics.aragog.allow_resume_law_change`` is set.

A record that names the Aragog module and lacks ``separation_viscosity`` counts as
aragog's ``'melt'``. Any other law the record does not hold must be set in the config
file of the resume; the allow switch does not stand in for a missing record.
"""

from __future__ import annotations

import logging
import os
import tomllib

log = logging.getLogger('fwl.' + __name__)

LAW_FIELDS = ('separation_viscosity', 'separation_buoyancy')
UNRECORDED_VISCOSITY = 'melt'  # aragog's default, which a record without the field ran under
# The permission to switch a separation law holds for one resume, so the record keeps it off.
RECORD_OVERRIDES = {'interior_energetics.aragog.allow_resume_law_change': False}


class ResumeSeparationLawError(RuntimeError):
    """A resume would change a separation law the run was written with."""


def _aragog_table(path) -> tuple[str | None, dict]:
    """Read ``path``; return its ``interior_energetics.module`` (None if absent or
    unreadable) and its ``interior_energetics.aragog`` table."""
    try:
        with open(path, 'rb') as f:
            raw = tomllib.load(f)
    except (OSError, ValueError):
        return None, {}
    interior = raw.get('interior_energetics')
    interior = interior if isinstance(interior, dict) else {}
    aragog = interior.get('aragog')
    return interior.get('module'), aragog if isinstance(aragog, dict) else {}


def _unrecorded(key: str, stored_path: str, module, config_path) -> str:
    """Refusal text for a law the run did not record."""
    if module is None:
        where = f'{stored_path}, which is missing, unreadable or names no interior module'
        hint = 'the law of the run is unknown'
    else:
        where = stored_path
        hint = (
            "a run without the field ran under 'solid' with an aragog below 26.10.22 and "
            "under 'mixture' from 26.10.22 (the 'aragog version' line of the run's "
            'proteus_NN.log files written before this attempt)'
        )
    return (
        f'{key} is not recorded in {where}: {hint}; set it in {config_path} to the law the '
        'run used (allow_resume_law_change does not cover a law the run did not record)'
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
        when the laws stay, and when the run or the resume does not use Aragog.

    Raises
    ------
    ResumeSeparationLawError
        If a law changes and ``allow_resume_law_change`` is false, or the run did not
        record a law and the config file does not set it. Nothing is logged then.
    """
    stored_path = os.path.join(output_dir, 'init_coupler.toml')
    module, stored = _aragog_table(stored_path)
    if config.interior_energetics.module != 'aragog' or module not in (None, 'aragog'):
        return []
    aragog = config.interior_energetics.aragog
    in_file = _aragog_table(config_path)[1]
    switches, refusals, notes = [], [], []
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
        if old == new:
            continue
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
                f'{key} is {new!r} in {config_path}, but the run was written with {old!r} '
                f'({source}); set it to {old!r} to keep the law, or set interior_energetics.'
                'aragog.allow_resume_law_change = true to switch it at the resume'
            )
    if refusals:
        raise ResumeSeparationLawError('Resume refused:\n  ' + '\n  '.join(refusals))
    for level, note in notes:
        log.log(level, note)
    return switches
