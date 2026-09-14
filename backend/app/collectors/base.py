"""The seam between "where telemetry comes from" and everything else.

`FleetSimulator` and `SNMPCollector` both satisfy this protocol, which is
what lets the simulator be swapped for real devices without the model,
explainer, alert engine or dashboard changing at all.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol

from ..simulator import Device, MetricSnapshot

logger = logging.getLogger(__name__)

DEFAULT_INVENTORY = Path("config/devices.json")


class Collector(Protocol):
    devices: list[Device]

    def step(self) -> list[MetricSnapshot]:
        """Produce one reading per device. May return fewer than `devices`
        when a device cannot be read this cycle."""


def build_collector(n_devices: int = 14, seed: int = 7):
    """Pick a collector from the environment.

    AURA_COLLECTOR=simulator (default) -- synthetic fleet, no hardware needed.
    AURA_COLLECTOR=snmp                -- poll real devices listed in
                                          $AURA_INVENTORY (default config/devices.json).
    """
    kind = os.environ.get("AURA_COLLECTOR", "simulator").strip().lower()

    if kind == "simulator":
        from ..simulator import FleetSimulator, build_fleet

        return FleetSimulator(build_fleet(n_devices=n_devices, seed=seed), seed=seed)

    if kind == "snmp":
        from .inventory import load_inventory
        from .snmp import PySnmpSession, SNMPCollector

        inventory_path = Path(os.environ.get("AURA_INVENTORY", DEFAULT_INVENTORY))
        targets = load_inventory(inventory_path)
        logger.info("SNMP collector: %d devices from %s", len(targets), inventory_path)
        return SNMPCollector(targets, session=PySnmpSession())

    raise ValueError(f"Unknown AURA_COLLECTOR={kind!r}; expected 'simulator' or 'snmp'")
