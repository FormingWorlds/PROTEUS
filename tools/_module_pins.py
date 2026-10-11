"""Read module pins from pyproject.toml.

Single entry point for tools/get_*.sh, install.sh and the CI composite
action to look up how an external module is pinned. Two kinds of pin
live in pyproject.toml:

- ``[tool.proteus.modules.<name>]`` holds the URL and ref of a module
  cloned outside pip (AGNI, SOCRATES, SPIDER, LovePy, ...), or the url,
  mirror and sha256 of the PETSc archive.
- ``[project] dependencies`` and ``[project.optional-dependencies]``
  hold the ``>=`` version floor of each Python distribution
  (``fwl-mors``, ``fwl-aragog``, ...). The installer checks out the
  release tag whose version equals that floor.

Usage from a shell script:

    URL=$(python tools/_module_pins.py agni url)
    REF=$(python tools/_module_pins.py agni ref)
    MIRROR=$(python tools/_module_pins.py petsc mirror)
    FLOOR=$(python tools/_module_pins.py fwl-mors floor)
    TAG=$(git -C MORS tag -l | python tools/_module_pins.py fwl-mors floor-tag)

``floor-tag`` reads tag names from standard input, one per line, and
prints the one tag whose version equals the floor: the number fields are
compared as integers, so ``26.10.06`` matches the floor ``26.10.6``.

Requirements are parsed with a small regex rather than ``packaging``:
the tool runs before PROTEUS is installed, and an environment holding
only python and pip has no importable ``packaging``.

Exits non-zero with a clear message if the module or distribution is
unknown, the requested field is missing, the distribution has no ``>=``
floor, or not exactly one tag matches the floor.
"""

from __future__ import annotations

import pathlib
import re
import sys
import tomllib

USAGE = (
    'usage: python tools/_module_pins.py <module> <field>\n'
    '         fields of [tool.proteus.modules.<module>]: url, ref, mirror, sha256\n'
    '       python tools/_module_pins.py <distribution> floor\n'
    '       python tools/_module_pins.py <distribution> floor-tag  < tag-list'
)
MAX_TAGS_SHOWN = 10

_REQUIREMENT = re.compile(r'^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(.*)$')
_SPECIFIER = re.compile(r'^(===|~=|==|!=|<=|>=|<|>)\s*(\S+)$')
_RELEASE = re.compile(r'^\d+(?:\.\d+)*$')


def _repo_root() -> pathlib.Path:
    """Return the PROTEUS repo root, identified by ``pyproject.toml``.

    Walks up from this file's directory so the helper works whether it
    is called from the repo root, ``tools/``, or a CI checkout copy.
    """
    here = pathlib.Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / 'pyproject.toml').is_file():
            return candidate
    raise SystemExit('Could not locate pyproject.toml above _module_pins.py')


def _normalise(name: str) -> str:
    """Return a distribution name in its PEP 503 normalised form.

    Parameters
    ----------
    name : str
        Distribution name as written, for example ``FWL_Mors``.

    Returns
    -------
    str
        Lower-case name with each run of ``-``, ``_`` and ``.`` replaced
        by one ``-``, for example ``fwl-mors``.
    """
    return re.sub(r'[-_.]+', '-', name).lower()


def _specifiers(text: str) -> list[tuple[str, str]]:
    """Return the version specifiers of a requirement string.

    Parameters
    ----------
    text : str
        A PEP 508 requirement such as ``fwl-mors>=26.10.6,<27``. An
        environment marker after ``;`` is dropped.

    Returns
    -------
    list of (str, str)
        The ``(operator, version)`` pairs, for example
        ``[('>=', '26.10.6'), ('<', '27')]``; empty for a requirement
        without a specifier.

    Raises
    ------
    ValueError
        If one of the specifiers cannot be parsed.
    """
    rest = _REQUIREMENT.match(text.split(';', 1)[0]).group(2).strip()
    if rest.startswith('(') and rest.endswith(')'):
        rest = rest[1:-1]
    found = []
    for part in filter(None, (p.strip() for p in rest.split(','))):
        spec = _SPECIFIER.match(part)
        if not spec:
            raise ValueError(f'cannot parse specifier {part!r} in requirement {text!r}')
        found.append((spec.group(1), spec.group(2)))
    return found


def _requirements_by_name(data: dict) -> dict[str, list[str]]:
    """Return the requirement strings in ``[project]``, keyed by distribution name.

    Parameters
    ----------
    data : dict
        Parsed ``pyproject.toml``.

    Returns
    -------
    dict of str to list of str
        The entries of ``dependencies`` and of every list in
        ``optional-dependencies``, under their normalised name.
    """
    project = data.get('project', {})
    texts = list(project.get('dependencies', []))
    for extra in project.get('optional-dependencies', {}).values():
        texts.extend(extra)
    found = {}
    for text in texts:
        match = _REQUIREMENT.match(text)
        if match:
            found.setdefault(_normalise(match.group(1)), []).append(text)
    return found


def find_floor(data: dict, distribution: str) -> str:
    """Return the ``>=`` version floor of a distribution.

    Parameters
    ----------
    data : dict
        Parsed ``pyproject.toml``.
    distribution : str
        Distribution name, for example ``fwl-mors``.

    Returns
    -------
    str
        The floor text as written, for example ``26.10.6``.

    Raises
    ------
    LookupError
        If the distribution is not required, has no single ``>=``
        specifier, or is required twice with different floors.
    """
    floors = {}
    for text in _requirements_by_name(data).get(_normalise(distribution), []):
        specifiers = _specifiers(text)
        lower = [version for op, version in specifiers if op == '>=']
        if len(lower) != 1:
            shown = ','.join(op + version for op, version in specifiers) or 'none'
            raise LookupError(
                f"'{distribution}' has no single >= version floor in pyproject.toml: "
                f'it is required as {text!r} (specifier: {shown})'
            )
        floors[lower[0]] = text
    if not floors:
        raise LookupError(f"'{distribution}' is not a dependency in pyproject.toml")
    if len(floors) > 1:
        raise LookupError(
            f"'{distribution}' is required with different floors: {', '.join(floors.values())}"
        )
    return next(iter(floors))


def _release_key(version: str) -> tuple[int, ...] | None:
    """Return the integer fields of a release number, trailing zeros removed.

    Parameters
    ----------
    version : str
        A version or tag such as ``26.10.06`` or ``v0.2.0``; one leading
        ``v`` is ignored.

    Returns
    -------
    tuple of int or None
        ``(26, 10, 6)`` for ``26.10.06`` and ``(0, 2)`` for ``v0.2.0``;
        None when anything other than digits and dots remains.
    """
    text = version.strip().removeprefix('v')
    if not _RELEASE.match(text):
        return None
    fields = [int(f) for f in text.split('.')]
    while len(fields) > 1 and fields[-1] == 0:
        fields.pop()
    return tuple(fields)


def match_floor_tag(distribution: str, floor: str, tags: list[str]) -> str:
    """Return the one tag whose release number equals the floor.

    Parameters
    ----------
    distribution : str
        Distribution name, used in the error message.
    floor : str
        Floor version text, for example ``26.10.6``.
    tags : list of str
        Tag names; a tag that is not a plain release number is skipped.

    Returns
    -------
    str
        The matching tag as written, for example ``26.10.06``.

    Raises
    ------
    LookupError
        If the floor is not a plain release number, or no tag or more
        than one tag matches it.
    """
    wanted = _release_key(floor)
    if wanted is None:
        raise LookupError(
            f"'{distribution}' floor {floor} is not a plain release number to match a tag"
        )
    candidates = {}
    for tag in (t.strip() for t in tags):
        key = _release_key(tag)
        if key is not None:
            candidates[tag] = key
    matches = sorted(tag for tag, key in candidates.items() if key == wanted)
    if len(matches) == 1:
        return matches[0]

    if matches:
        problem = f'{len(matches)} tags match it: {", ".join(matches)}'
    elif candidates:
        highest = sorted(candidates, key=candidates.get, reverse=True)[:MAX_TAGS_SHOWN]
        problem = (
            f'no tag matches it; release tags seen: {len(candidates)}, '
            f'highest: {", ".join(highest)}'
        )
    else:
        problem = 'no tag matches it; saw no release tags'
    raise LookupError(f"'{distribution}' floor {floor}: {problem}")


def _module_field(data: dict, module: str, field: str) -> str:
    """Return one field of ``[tool.proteus.modules.<module>]``.

    Parameters
    ----------
    data : dict
        Parsed ``pyproject.toml``.
    module : str
        Module table name, for example ``agni``.
    field : str
        Field name, for example ``ref``.

    Returns
    -------
    str
        The field value.

    Raises
    ------
    LookupError
        If the module or the field is missing.
    """
    modules = data.get('tool', {}).get('proteus', {}).get('modules', {})
    if module not in modules:
        if _normalise(module) in _requirements_by_name(data):
            raise LookupError(
                f"'{module}' is a distribution in pyproject.toml; "
                'the fields of a distribution are floor and floor-tag'
            )
        known = ', '.join(sorted(modules)) or '(none configured)'
        raise LookupError(f"Unknown module '{module}'. Known: {known}")
    entry = modules[module]
    if field not in entry:
        raise LookupError(
            f"Module '{module}' has no field '{field}'. Available: {', '.join(sorted(entry))}"
        )
    return str(entry[field])


def main(argv: list[str]) -> int:
    """Print the requested pin and return the process exit status.

    Parameters
    ----------
    argv : list of str
        ``sys.argv``: the program name, a module or distribution name,
        and a field.

    Returns
    -------
    int
        0 on success, 1 when the pin cannot be resolved, 2 on a usage
        error.
    """
    if len(argv) != 3:
        print(USAGE, file=sys.stderr)
        return 2

    _, name, field = argv
    path = _repo_root() / 'pyproject.toml'
    try:
        data = tomllib.loads(path.read_text())
        if field == 'floor':
            value = find_floor(data, name)
        elif field == 'floor-tag':
            value = match_floor_tag(name, find_floor(data, name), sys.stdin.read().splitlines())
        else:
            value = _module_field(data, name, field)
    except tomllib.TOMLDecodeError as err:
        print(f'cannot parse {path}: {err}', file=sys.stderr)
        return 1
    except (LookupError, ValueError) as err:
        print(err.args[0], file=sys.stderr)
        return 1

    print(value)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
