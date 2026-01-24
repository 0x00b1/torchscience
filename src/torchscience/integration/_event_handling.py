"""Event handling for ODE integration.

Events are zero-crossings of user-defined functions g(t, y).
When g changes sign, root-finding locates the exact crossing time.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch


def _to_scalar(value):
    """Convert tensor to scalar if needed."""
    return value.item() if isinstance(value, torch.Tensor) else value


@dataclass
class EventSpec:
    """Specification for an event function."""

    func: Callable[[Union[float, torch.Tensor], torch.Tensor], torch.Tensor]
    terminal: bool = False
    direction: int = 0  # 0=both, 1=positive, -1=negative


def parse_events(events: Optional[List[Callable]]) -> List[EventSpec]:
    """Parse event functions into EventSpec objects."""
    if events is None:
        return []

    specs = []
    for func in events:
        terminal = getattr(func, "terminal", False)
        direction = getattr(func, "direction", 0)
        if direction not in {-1, 0, 1}:
            raise ValueError(
                f"Event direction must be -1, 0, or 1, got {direction}"
            )
        specs.append(
            EventSpec(func=func, terminal=terminal, direction=direction)
        )

    return specs


def detect_event(
    event: EventSpec,
    t0: float,
    y0: torch.Tensor,
    t1: float,
    y1: torch.Tensor,
) -> bool:
    """Check if event occurred between (t0, y0) and (t1, y1)."""
    g0 = _to_scalar(event.func(t0, y0))
    g1 = _to_scalar(event.func(t1, y1))

    # Check for sign change (explicit comparison avoids overflow issues)
    if (g0 > 0 and g1 > 0) or (g0 < 0 and g1 < 0):
        return False

    # Handle exact zero
    if g0 == 0 and g1 == 0:
        return False

    # Check direction
    if event.direction > 0 and g1 <= g0:
        return False
    if event.direction < 0 and g1 >= g0:
        return False

    return True


def locate_event(
    event: EventSpec,
    t0: float,
    y0: torch.Tensor,
    t1: float,
    y1: torch.Tensor,
    interp: Callable,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> Tuple[float, torch.Tensor]:
    """
    Locate event time using bisection with interpolant.

    Returns
    -------
    t_event : float
        Time of event.
    y_event : Tensor
        State at event.
    """
    g0 = _to_scalar(event.func(t0, y0))

    for _ in range(max_iter):
        t_mid = (t0 + t1) / 2
        y_mid = interp(t_mid)
        g_mid = _to_scalar(event.func(t_mid, y_mid))

        if abs(g_mid) < tol or (t1 - t0) < tol:
            return t_mid, y_mid

        if g0 * g_mid < 0:
            t1 = t_mid
        else:
            t0 = t_mid
            g0 = g_mid

    t_final = (t0 + t1) / 2
    return t_final, interp(t_final)


class EventTracker:
    """Tracks events during integration."""

    def __init__(self, events: List[EventSpec]):
        self.events = events
        self.t_events: List[List[float]] = [[] for _ in events]
        self.y_events: List[List[torch.Tensor]] = [[] for _ in events]
        self.terminated = False
        self.termination_time: Optional[float] = None

    def check_and_handle(
        self,
        t0: float,
        y0: torch.Tensor,
        t1: float,
        y1: torch.Tensor,
        interp: Callable,
    ) -> Optional[Tuple[float, torch.Tensor]]:
        """
        Check for events and handle them.

        Returns (t_event, y_event) if a terminal event occurred, else None.
        """
        for i, event in enumerate(self.events):
            if detect_event(event, t0, y0, t1, y1):
                t_event, y_event = locate_event(event, t0, y0, t1, y1, interp)
                self.t_events[i].append(t_event)
                self.y_events[i].append(y_event)

                if event.terminal:
                    self.terminated = True
                    self.termination_time = t_event
                    return t_event, y_event

        return None
