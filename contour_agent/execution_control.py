"""Optional cooperative execution limits, isolated to the current context.

Deadlines are absolute values of the supplied monotonic clock. Nested scopes
inherit every cancellation predicate and the earliest deadline for each clock;
distinct clocks retain their own domains. No scope means no extra work or limit.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
from numbers import Real
import time
from typing import Callable, Iterator


class ExecutionInterrupted(InterruptedError):
    """A cancellation or deadline with a bounded, non-sensitive diagnostic."""

    def __init__(self, reason: str):
        if reason not in {"cancelled", "deadline"}:
            raise ValueError("Unsupported execution interruption reason")
        self.reason = reason
        super().__init__("Execution interrupted: " + reason)


@dataclass(frozen=True)
class _Scope:
    cancelled: tuple[Callable[[], bool], ...]
    deadlines: tuple[tuple[float, Callable[[], float]], ...]


_CURRENT_SCOPE: ContextVar[_Scope | None] = ContextVar("cad_execution_scope", default=None)


@contextmanager
def execution_scope(*, cancelled: Callable[[], bool] | None = None,
                    deadline: float | None = None,
                    clock: Callable[[], float] = time.monotonic) -> Iterator[None]:
    """Install optional limits without changing existing numerical budgets.

    The context is reset even when work raises. New threads start with no scope;
    a worker that needs job controls must enter its own scope. Checks are
    explicit, so entering a scope itself never starts or cancels any work.
    """
    if cancelled is not None and not callable(cancelled):
        raise TypeError("Execution cancellation must be a callable or None")
    if not callable(clock):
        raise TypeError("Execution clock must be callable")
    if deadline is not None and (isinstance(deadline, bool) or
            not isinstance(deadline, Real) or not math.isfinite(float(deadline))):
        raise ValueError("Execution deadline must be finite or None")
    parent = _CURRENT_SCOPE.get()
    predicates = list(parent.cancelled) if parent is not None else []
    deadlines = list(parent.deadlines) if parent is not None else []
    if cancelled is not None and not any(item is cancelled for item in predicates):
        predicates.append(cancelled)
    if deadline is not None:
        value = float(deadline)
        for index, (inherited, inherited_clock) in enumerate(deadlines):
            if inherited_clock is clock:
                deadlines[index] = (min(inherited, value), clock)
                break
        else:
            deadlines.append((value, clock))
    token = _CURRENT_SCOPE.set(_Scope(tuple(predicates), tuple(deadlines)))
    try:
        yield
    finally:
        _CURRENT_SCOPE.reset(token)


def execution_checkpoint(stage: str | None = None) -> None:
    """Raise safely when a current-context limit is reached.

    ``stage`` labels call sites for readers and future instrumentation only.
    It is deliberately not retained in exceptions, avoiding accidental secret
    exposure from a caller-supplied diagnostic. Cancellation wins over deadline
    when both are observed at the same checkpoint.
    """
    scope = _CURRENT_SCOPE.get()
    if scope is None:
        return
    for cancelled in scope.cancelled:
        if cancelled():
            raise ExecutionInterrupted("cancelled")
    for deadline, clock in scope.deadlines:
        now = clock()
        if isinstance(now, bool) or not isinstance(now, Real) or not math.isfinite(float(now)):
            raise ValueError("Execution clock must return a finite time")
        if float(now) >= deadline:
            raise ExecutionInterrupted("deadline")
