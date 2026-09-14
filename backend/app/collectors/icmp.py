"""Active ICMP probing for latency, jitter and packet loss.

SNMP does not expose these three: a device will happily report its own CPU
and interface counters while the path to it is badly degraded. They have to
be measured actively, by probing.

This wraps the system `ping` binary rather than opening raw sockets, because
raw ICMP sockets require root (or CAP_NET_RAW) and a monitoring daemon
should not need that. The trade-off is that output parsing depends on the
local ping implementation, so both the Linux (iputils) and BSD/macOS output
formats are handled, and `parse_ping_output` is kept separate from the
subprocess call so it can be tested against captured output.

In production you would more likely drive these probes from a dedicated
probe agent, or read Cisco IP SLA / Juniper RPM results over SNMP, so that
you measure the path a user's traffic actually takes rather than the path
from the monitoring host.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

# "5 packets transmitted, 3 received, 40% packet loss, time 4052ms"   (iputils)
# "5 packets transmitted, 3 packets received, 40.0% packet loss"      (BSD)
_LOSS_RE = re.compile(r"([\d.]+)%\s+packet loss")

# "rtt min/avg/max/mdev = 12.301/12.456/12.678/0.134 ms"              (iputils)
# "round-trip min/avg/max/stddev = 12.301/12.456/12.678/0.134 ms"     (BSD)
_RTT_RE = re.compile(
    r"(?:rtt|round-trip)\s+min/avg/max/(?:mdev|stddev)\s*=\s*"
    r"([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)"
)


@dataclass
class ProbeResult:
    packet_loss_pct: float
    latency_ms: Optional[float]  # None when every probe was lost
    jitter_ms: Optional[float]

    @property
    def reachable(self) -> bool:
        return self.packet_loss_pct < 100.0


def parse_ping_output(output: str) -> Optional[ProbeResult]:
    """Parse ping summary output. Returns None if it isn't recognisable."""
    loss_match = _LOSS_RE.search(output)
    if loss_match is None:
        return None

    loss = float(loss_match.group(1))
    rtt_match = _RTT_RE.search(output)
    if rtt_match is None:
        # No RTT line: everything was lost, or the host never answered.
        return ProbeResult(packet_loss_pct=loss, latency_ms=None, jitter_ms=None)

    return ProbeResult(
        packet_loss_pct=loss,
        latency_ms=float(rtt_match.group(2)),  # avg
        jitter_ms=float(rtt_match.group(4)),   # mdev / stddev
    )


def ping_available() -> bool:
    return shutil.which("ping") is not None


def probe(host: str, count: int = 5, timeout_seconds: int = 2) -> Optional[ProbeResult]:
    """Probe `host`. Returns None when the probe could not be run or parsed.

    A fully unreachable host is NOT None -- it is a valid 100%-loss result,
    which is exactly the signal the failure predictor wants to see.
    """
    if not ping_available():
        return None

    try:
        completed = subprocess.run(
            ["ping", "-c", str(count), "-W", str(timeout_seconds), host],
            capture_output=True,
            text=True,
            # generous: ping itself bounds the runtime via -c/-W
            timeout=count * timeout_seconds + 10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None

    # ping exits non-zero when packets are lost, and that output still
    # carries the statistics we want, so the return code is not checked.
    return parse_ping_output(completed.stdout)
