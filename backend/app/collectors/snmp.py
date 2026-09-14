"""SNMP collector: polls real devices and emits MetricSnapshot objects.

This is the drop-in replacement for `FleetSimulator` -- it produces the same
`list[MetricSnapshot]` from `step()`, so the feature pipeline, model,
explainer, alert engine and dashboard all work unchanged.

**Status: scaffold.** The counter maths, snapshot assembly and inventory
handling are covered by tests, but the SNMP transport itself has not been
run against real hardware. Verify against one device before trusting it
against a fleet. The transport is deliberately isolated behind the
`SnmpSession` protocol so the rest of the pipeline can be tested (and so you
can swap pysnmp for easysnmp, a Prometheus scrape, or a NetBox-driven
poller) without touching the logic above it.

Two honest caveats about real data:

  * `failure_event` is always False here. The simulator knows ground truth;
    a real collector cannot. Training labels have to come from somewhere
    else -- syslog link-down events, ticket history, device up/down records.
    Until you have those, run this in collect-only mode (no trained model
    present) and just accumulate telemetry.

  * `device_state` is inferred from reachability alone, not from the rich
    state machine the simulator provides.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Optional, Protocol

from ..simulator import Device, MetricSnapshot
from . import icmp
from .counters import CounterTracker, per_minute, utilisation_pct
from .inventory import DeviceTarget

logger = logging.getLogger(__name__)

# An interface cannot realistically produce more than this many errors per
# second; beyond it, a counter drop is a reset rather than a wrap.
MAX_PLAUSIBLE_ERRORS_PER_SECOND = 1_000_000
# 400 Gbit/s in octets/s, a ceiling no campus link will exceed.
MAX_PLAUSIBLE_OCTETS_PER_SECOND = 50_000_000_000

# Stand-ins used only for a device that has never answered a probe, so that a
# device down since start-up still reaches the dashboard. Chosen to match the
# explainer's critical thresholds, i.e. to read as bad rather than plausible.
UNREACHABLE_LATENCY_MS = 50.0
UNREACHABLE_JITTER_MS = 15.0

# Without these there is nothing worth recording, so the device is skipped.
REQUIRED_METRICS = (
    "latency_ms",
    "packet_loss_pct",
    "jitter_ms",
    "bandwidth_util_pct",
    "interface_errors_per_min",
)

# CPU, memory and temperature OIDs are vendor-specific and absent on plenty
# of hardware; tcpRetransSegs reflects only the device's own TCP stack.
# Requiring them would mean collecting nothing at all from such devices, so
# they fall back to a fixed value instead. A *constant* feature is the right
# behaviour here: its slope, std and delta are all zero, so the trend
# features the model relies on carry no signal from a metric the device
# cannot report. If most of your fleet lacks one of these, prefer retraining
# without that metric over feeding a constant.
OPTIONAL_METRIC_DEFAULTS = {
    "cpu_pct": 0.0,
    "memory_pct": 0.0,
    "temperature_c": 0.0,
    "retransmits_per_min": 0.0,
}


class SnmpSession(Protocol):
    """Minimal transport contract, so the poller can be faked in tests."""

    def get(self, target: DeviceTarget, oids: dict[str, str]) -> dict[str, Optional[float]]:
        """Return {name: value} for scalar OIDs. Missing/failed -> None."""

    def get_interface(
        self, target: DeviceTarget, if_index: int, oids: dict[str, str]
    ) -> dict[str, Optional[float]]:
        """Return {name: value} for per-interface OIDs at `if_index`."""


class PySnmpSession:
    """Real SNMP transport. Requires `pip install -r requirements-snmp.txt`.

    Untested against hardware -- see the module docstring.
    """

    SCALAR_KEYS = ("sysUpTime", "cpu_pct", "memory_used", "memory_free",
                   "temperature_c", "tcpRetransSegs")
    INTERFACE_KEYS = ("ifInErrors", "ifOutErrors", "ifInDiscards",
                      "ifHCInOctets", "ifHCOutOctets", "ifHighSpeed")

    def __init__(self, timeout_seconds: int = 2, retries: int = 1):
        try:
            from pysnmp import hlapi  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "SNMP collection needs pysnmp: pip install -r requirements-snmp.txt"
            ) from exc
        self.timeout = timeout_seconds
        self.retries = retries

    def get(self, target: DeviceTarget, oids: dict[str, str]) -> dict[str, Optional[float]]:
        wanted = {k: oids[k] for k in self.SCALAR_KEYS if k in oids}
        return self._fetch(target, wanted)

    def get_interface(
        self, target: DeviceTarget, if_index: int, oids: dict[str, str]
    ) -> dict[str, Optional[float]]:
        wanted = {k: f"{oids[k]}.{if_index}" for k in self.INTERFACE_KEYS if k in oids}
        return self._fetch(target, wanted)

    def _fetch(self, target: DeviceTarget, oids: dict[str, str]) -> dict[str, Optional[float]]:
        from pysnmp import hlapi

        community = target.community()
        if not community:
            raise RuntimeError(
                f"No SNMP community for {target.device_id}: set ${target.community_env}"
            )

        results: dict[str, Optional[float]] = {}
        # mpModel 0 = SNMPv1, 1 = SNMPv2c. SNMPv3 would use UsmUserData here.
        auth = hlapi.CommunityData(community, mpModel=0 if target.snmp_version == "1" else 1)

        for name, oid in oids.items():
            try:
                error_indication, error_status, _, var_binds = next(
                    hlapi.getCmd(
                        hlapi.SnmpEngine(),
                        auth,
                        hlapi.UdpTransportTarget(
                            (target.host, 161), timeout=self.timeout, retries=self.retries
                        ),
                        hlapi.ContextData(),
                        hlapi.ObjectType(hlapi.ObjectIdentity(oid)),
                    )
                )
                if error_indication or error_status:
                    logger.debug("SNMP %s %s: %s%s", target.host, name,
                                 error_indication or "", error_status or "")
                    results[name] = None
                else:
                    results[name] = float(var_binds[0][1])
            except Exception:
                logger.debug("SNMP %s %s failed", target.host, name, exc_info=True)
                results[name] = None
        return results


class SNMPCollector:
    """Polls a fleet over SNMP + ICMP and emits MetricSnapshot objects."""

    def __init__(
        self,
        targets: list[DeviceTarget],
        session: SnmpSession,
        probe: Callable[[str], Optional[icmp.ProbeResult]] = icmp.probe,
        clock: Callable[[], datetime] = datetime.utcnow,
    ):
        self.targets = targets
        self.session = session
        self.probe = probe
        self.clock = clock
        self.devices = [
            Device(
                device_id=t.device_id,
                name=t.name,
                device_type=t.device_type,
                site=t.site,
            )
            for t in targets
        ]
        self._errors = CounterTracker(32, MAX_PLAUSIBLE_ERRORS_PER_SECOND)
        self._octets = CounterTracker(64, MAX_PLAUSIBLE_OCTETS_PER_SECOND)
        self._retrans = CounterTracker(32, MAX_PLAUSIBLE_ERRORS_PER_SECOND)
        self._last_rtt: dict[str, tuple[float, float]] = {}

    def step(self) -> list[MetricSnapshot]:
        """Poll every device once. Devices that cannot be read are omitted --
        a gap is safer than a fabricated reading."""
        now = self.clock()
        snapshots = []
        for target in self.targets:
            try:
                snapshot = self._poll(target, now)
            except Exception:
                logger.warning("polling %s failed", target.device_id, exc_info=True)
                continue
            if snapshot is not None:
                snapshots.append(snapshot)
        return snapshots

    def _poll(self, target: DeviceTarget, now: datetime) -> Optional[MetricSnapshot]:
        oids = target.oids()
        scalars = self.session.get(target, oids)
        uptime = scalars.get("sysUpTime")
        uptime_ticks = int(uptime) if uptime is not None else None

        reachability = self.probe(target.host)
        if reachability is None:
            # Probe unavailable (no ping binary, or it failed to run). We can
            # still use SNMP values, but latency/jitter/loss are unknown.
            latency, jitter, loss = None, None, None
            link_up = any(v is not None for v in scalars.values())
        else:
            loss = reachability.packet_loss_pct
            link_up = reachability.reachable
            latency, jitter = self._rtt_for(target.device_id, reachability)

        interface = self._poll_interfaces(target, oids, now, uptime_ticks)

        values = {
            "latency_ms": latency,
            "packet_loss_pct": loss,
            "jitter_ms": jitter,
            "bandwidth_util_pct": interface["utilisation_pct"],
            "cpu_pct": scalars.get("cpu_pct"),
            "memory_pct": _memory_pct(scalars),
            "interface_errors_per_min": per_minute(interface["errors_per_second"]),
            "temperature_c": scalars.get("temperature_c"),
            "retransmits_per_min": self._retransmit_rate(target, scalars, now, uptime_ticks),
        }

        missing_required = [
            name for name in REQUIRED_METRICS if values.get(name) is None
        ]
        if missing_required:
            # Expected on the first poll: counters need two readings before a
            # rate exists. Persistent gaps mean an OID is wrong for this device.
            logger.debug(
                "skipping %s, no value for: %s", target.device_id, ", ".join(missing_required)
            )
            return None

        for name, fallback in OPTIONAL_METRIC_DEFAULTS.items():
            if values[name] is None:
                values[name] = fallback

        return MetricSnapshot(
            device_id=target.device_id,
            timestamp=now,
            link_up=link_up,
            device_state="normal" if link_up else "failed",
            active_fault="unknown",   # no ground truth on real hardware
            failure_event=False,      # labels must come from syslog/tickets
            **{k: round(float(v), 3) for k, v in values.items()},
        )

    def _poll_interfaces(
        self,
        target: DeviceTarget,
        oids: dict,
        now: datetime,
        uptime_ticks: Optional[int],
    ) -> dict:
        """Aggregate across the polled interfaces: errors summed (any
        interface erroring matters), utilisation taken from the busiest."""
        total_errors = None
        peak_utilisation = None

        for if_index in target.interfaces:
            raw = self.session.get_interface(target, if_index, oids)

            for key in ("ifInErrors", "ifOutErrors", "ifInDiscards"):
                if raw.get(key) is None:
                    continue
                rate = self._errors.update(
                    (target.device_id, key, if_index), int(raw[key]), now, uptime_ticks
                )
                if rate is not None:
                    total_errors = (total_errors or 0.0) + rate

            octets = None
            for key in ("ifHCInOctets", "ifHCOutOctets"):
                if raw.get(key) is None:
                    continue
                rate = self._octets.update(
                    (target.device_id, key, if_index), int(raw[key]), now, uptime_ticks
                )
                if rate is not None:
                    octets = (octets or 0.0) + rate

            speed_mbit = raw.get("ifHighSpeed")
            if octets is not None and speed_mbit:
                utilisation = utilisation_pct(octets, float(speed_mbit) * 1_000_000)
                peak_utilisation = max(peak_utilisation or 0.0, utilisation)

        return {"errors_per_second": total_errors, "utilisation_pct": peak_utilisation}


    def _retransmit_rate(
        self,
        target: DeviceTarget,
        scalars: dict,
        now: datetime,
        uptime_ticks: Optional[int],
    ) -> Optional[float]:
        raw = scalars.get("tcpRetransSegs")
        if raw is None:
            return None
        return per_minute(
            self._retrans.update(
                (target.device_id, "tcpRetransSegs"), int(raw), now, uptime_ticks
            )
        )

    def _rtt_for(
        self, device_id: str, reachability: icmp.ProbeResult
    ) -> tuple[float, float]:
        """Latency/jitter for this poll.

        A fully unreachable device returns no RTT at all, but dropping it
        would hide the single most important event this system exists to
        surface. So when every probe is lost we fall back to the last values
        seen for that device, or -- if it has never answered -- to the
        thresholds that read as unmistakably bad. The 100% loss figure
        alongside is the real signal; these keep the feature vector complete
        rather than pretending to be measurements.
        """
        if reachability.latency_ms is not None and reachability.jitter_ms is not None:
            rtt = (reachability.latency_ms, reachability.jitter_ms)
            self._last_rtt[device_id] = rtt
            return rtt
        return self._last_rtt.get(device_id, (UNREACHABLE_LATENCY_MS, UNREACHABLE_JITTER_MS))


def _memory_pct(scalars: dict) -> Optional[float]:
    used, free = scalars.get("memory_used"), scalars.get("memory_free")
    if used is None or free is None:
        return None
    total = used + free
    return (used / total * 100.0) if total > 0 else None
