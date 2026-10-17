"""Run-row physical invariant assertions and enforcement modes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from proteus.utils.helper import UpdateStatusfile

log = logging.getLogger('fwl.' + __name__)

STATUS_ERROR_ENERGY_INVARIANT: int = 30
STATUS_ERROR_MASS_INVARIANT: int = 31
STATUS_ERROR_STATE_BOUNDS_INVARIANT: int = 32


@dataclass(frozen=True)
class InvariantViolation:
    """Single run-row invariant violation record.

    Parameters
    ----------
    name : str
        Identifier of the invariant or helpfile column that failed.
    status_code : int
        PROTEUS status code (30 for energy, 31 for mass/structure, 32 for state bounds).
    value : float
        Observed residual or out-of-bounds value.
    tolerance : float
        Configured tolerance or bound threshold.
    message : str
        Diagnostic message formatting both the observed value and threshold.
    report_only : bool, default False
        When True, the violation logs a warning and never stops the run even when
        ``config.invariants.mode == 'stop'``.
    """

    name: str
    status_code: int
    value: float
    tolerance: float
    message: str
    report_only: bool = False


class InvariantViolationError(RuntimeError):
    """Raised when a stopping run-row invariant fails under ``mode == 'stop'``.

    Parameters
    ----------
    primary : InvariantViolation
        First stopping violation that triggered run termination.
    violations : Sequence[InvariantViolation], optional
        All stopping violations observed on the same row.
    """

    def __init__(
        self,
        primary: InvariantViolation,
        violations: Sequence[InvariantViolation] | None = None,
    ) -> None:
        self.status_code: int = primary.status_code
        self.invariant_name: str = primary.name
        self.value: float = primary.value
        self.tolerance: float = primary.tolerance
        self.violations: list[InvariantViolation] = (
            list(violations) if violations is not None else [primary]
        )
        joined = '; '.join(v.message for v in self.violations)
        super().__init__(
            f'Run-row invariant violation (status {self.status_code}, '
            f'{self.invariant_name}): {joined}'
        )


def handle_violations(
    violations: Sequence[InvariantViolation],
    mode: str = 'report',
    directories: dict | None = None,
) -> list[InvariantViolation]:
    """Process run-row invariant violations according to the configured mode.

    Parameters
    ----------
    violations : Sequence[InvariantViolation]
        Violations detected on the current helpfile row.
    mode : str, default 'report'
        Enforcement mode: ``'off'``, ``'report'``, or ``'stop'``.
    directories : dict or None, optional
        PROTEUS run directories dictionary; when provided and a stopping violation
        occurs under ``mode == 'stop'``, writes ``primary.status_code`` to the run
        status file before raising :class:`InvariantViolationError`.

    Returns
    -------
    list[InvariantViolation]
        List of processed violations (empty when ``mode == 'off'``).

    Raises
    ------
    InvariantViolationError
        When ``mode == 'stop'`` and at least one violation has ``report_only=False``.
    """
    if mode == 'off' or not violations:
        return []

    if mode == 'report':
        for v in violations:
            log.warning('Invariant report [%s]: %s', v.name, v.message)
        return list(violations)

    if mode != 'stop':
        raise ValueError(
            f"Unknown invariants.mode {mode!r}; expected 'off', 'report', or 'stop'."
        )

    stopping: list[InvariantViolation] = []
    for v in violations:
        if v.report_only:
            log.warning('Invariant report [%s]: %s', v.name, v.message)
        else:
            log.error('Invariant violation [%s]: %s', v.name, v.message)
            stopping.append(v)

    if stopping:
        primary = stopping[0]
        if directories is not None and 'output' in directories:
            UpdateStatusfile(directories, primary.status_code)
        raise InvariantViolationError(primary, stopping)

    return list(violations)
