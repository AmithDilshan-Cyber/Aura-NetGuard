"""Turning SNMP counters into rates, safely.

SNMP error/octet counters (ifInErrors, ifHCInOctets, ...) are monotonically
increasing totals, not rates. Naively subtracting consecutive polls produces
two classic false-alert sources:

  1. **Counter wrap.** A Counter32 rolls over at 2^32 back to 0. The naive
     delta goes negative; a naive abs() produces a huge fake spike.

  2. **Device reboot.** The counter resets to 0. This looks exactly like a
     wrap, but the correct response is the opposite: a wrap should be
     accounted for, a reset should be *discarded*, because the counts before
     and after are not comparable.

Telling them apart is what this module is for. Two signals are used:

  - `sysUpTime` (authoritative): if a device's uptime went backwards, it
    rebooted, so the counter reset. Always poll it alongside the counters.
  - Rate plausibility (fallback, when uptime is unavailable): if treating
    the drop as a wrap implies a rate that the interface physically cannot
    produce, it was a reset, not a wrap.

When a rate cannot be trusted, `update()` returns None rather than a
guessed number. The caller should skip that sample: a gap in the data is
recoverable, a fabricated spike wakes someone at 3am for nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

COUNTER32_MAX = 2**32
COUNTER64_MAX = 2**64


@dataclass
class _Sample:
    value: int
    timestamp: datetime
    uptime_ticks: Optional[int]


class CounterTracker:
    """Per-key rate tracker for monotonic counters.

    A key is whatever identifies one counter on one device, e.g.
    ("dev-001", "ifInErrors", "Gi0/1").
    """

    def __init__(self, width: int = 32, max_plausible_rate: Optional[float] = None):
        if width not in (32, 64):
            raise ValueError("SNMP counters are either Counter32 or Counter64")
        self.max_value = COUNTER32_MAX if width == 32 else COUNTER64_MAX
        self.max_plausible_rate = max_plausible_rate
        self._last: dict[tuple, _Sample] = {}

    def reset(self, key: tuple) -> None:
        self._last.pop(key, None)

    def update(
        self,
        key: tuple,
        value: int,
        timestamp: datetime,
        uptime_ticks: Optional[int] = None,
    ) -> Optional[float]:
        """Record a counter reading and return the rate per second since the
        previous reading, or None when no trustworthy rate can be derived.

        `uptime_ticks` is SNMP sysUpTime in hundredths of a second.
        """
        previous = self._last.get(key)
        self._last[key] = _Sample(value, timestamp, uptime_ticks)

        if previous is None:
            return None  # first reading establishes a baseline only

        elapsed = (timestamp - previous.timestamp).total_seconds()
        if elapsed <= 0:
            return None  # clock went backwards, or a duplicate poll

        if self._rebooted(previous, uptime_ticks):
            return None

        if value >= previous.value:
            delta = value - previous.value
        else:
            # A drop is either a wrap or a reset. Assume a wrap, then let the
            # plausibility check below reject it if that implies an
            # impossible rate -- in which case it was a reset.
            delta = (self.max_value - previous.value) + value

        if self._implausible(delta, elapsed):
            return None
        return delta / elapsed

    def _rebooted(self, previous: _Sample, uptime_ticks: Optional[int]) -> bool:
        if uptime_ticks is None or previous.uptime_ticks is None:
            return False
        return uptime_ticks < previous.uptime_ticks

    def _implausible(self, delta: int, elapsed: float) -> bool:
        if self.max_plausible_rate is None:
            return False
        return (delta / elapsed) > self.max_plausible_rate


def utilisation_pct(octets_per_second: float, link_speed_bits_per_second: float) -> float:
    """Interface octet rate -> percentage of link capacity."""
    if link_speed_bits_per_second <= 0:
        return 0.0
    pct = (octets_per_second * 8.0) / link_speed_bits_per_second * 100.0
    return max(0.0, min(100.0, pct))


def per_minute(rate_per_second: Optional[float]) -> Optional[float]:
    return None if rate_per_second is None else rate_per_second * 60.0
