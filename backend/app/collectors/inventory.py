"""Device inventory: what to poll, where, and with which OIDs.

Credentials are deliberately NOT stored here. The inventory file names an
environment variable per device (or a shared default), so the file itself
stays safe to commit and share; the secrets live in the environment.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..simulator import DEVICE_TYPES

# Standard IF-MIB / SNMPv2-MIB objects, the same on every vendor.
STANDARD_OIDS = {
    "sysUpTime": "1.3.6.1.2.1.1.3.0",
    "ifInErrors": "1.3.6.1.2.1.2.2.1.14",     # Counter32, per-interface
    "ifOutErrors": "1.3.6.1.2.1.2.2.1.20",    # Counter32, per-interface
    "ifInDiscards": "1.3.6.1.2.1.2.2.1.13",   # Counter32, per-interface
    "ifHCInOctets": "1.3.6.1.2.1.31.1.1.1.6",   # Counter64, per-interface
    "ifHCOutOctets": "1.3.6.1.2.1.31.1.1.1.10", # Counter64, per-interface
    "ifHighSpeed": "1.3.6.1.2.1.31.1.1.1.15",   # Mbit/s, per-interface
    # NOTE: this counts the device's OWN TCP stack, not traffic it forwards,
    # so on a router or switch it says very little. Kept because the model
    # expects the field; expect it to carry almost no signal on real gear.
    "tcpRetransSegs": "1.3.6.1.2.1.6.12.0",
}

# CPU/memory/temperature OIDs are vendor-specific -- there is no universal
# standard. These are the common Cisco ones as a starting point; check your
# own hardware's MIB and override per device profile in the inventory file.
VENDOR_OID_PROFILES = {
    "cisco": {
        "cpu_pct": "1.3.6.1.4.1.9.9.109.1.1.1.1.7.1",    # cpmCPUTotal5minRev
        "memory_used": "1.3.6.1.4.1.9.9.48.1.1.1.5.1",   # ciscoMemoryPoolUsed
        "memory_free": "1.3.6.1.4.1.9.9.48.1.1.1.6.1",   # ciscoMemoryPoolFree
        "temperature_c": "1.3.6.1.4.1.9.9.13.1.3.1.3.1",  # ciscoEnvMonTemperature
    },
    # Add your own: mikrotik, juniper, hp, ubiquiti ...
    "generic": {},
}


class InventoryError(ValueError):
    pass


@dataclass
class DeviceTarget:
    device_id: str
    name: str
    device_type: str
    site: str
    host: str
    interfaces: list[int] = field(default_factory=list)  # ifIndex values to poll
    snmp_version: str = "2c"
    oid_profile: str = "cisco"
    community_env: str = "AURA_SNMP_COMMUNITY"
    oid_overrides: dict = field(default_factory=dict)

    def community(self) -> Optional[str]:
        """Read the SNMP community string from the environment."""
        return os.environ.get(self.community_env)

    def oids(self) -> dict:
        profile = dict(VENDOR_OID_PROFILES.get(self.oid_profile, {}))
        profile.update(self.oid_overrides)
        return {**STANDARD_OIDS, **profile}


def load_inventory(path: Path) -> list[DeviceTarget]:
    if not path.exists():
        raise InventoryError(
            f"No inventory at {path}. Copy config/devices.example.json and edit it."
        )

    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise InventoryError(f"{path} is not valid JSON: {exc}") from exc

    devices = raw.get("devices") if isinstance(raw, dict) else raw
    if not isinstance(devices, list) or not devices:
        raise InventoryError(f"{path} contains no devices")

    targets = [_parse_device(entry, index, path) for index, entry in enumerate(devices)]

    seen: set[str] = set()
    for target in targets:
        if target.device_id in seen:
            raise InventoryError(f"duplicate device_id {target.device_id!r} in {path}")
        seen.add(target.device_id)
    return targets


def _parse_device(entry: dict, index: int, path: Path) -> DeviceTarget:
    for required in ("device_id", "host", "device_type"):
        if not entry.get(required):
            raise InventoryError(f"device #{index} in {path} is missing {required!r}")

    device_type = entry["device_type"]
    if device_type not in DEVICE_TYPES:
        raise InventoryError(
            f"device {entry['device_id']!r} has device_type {device_type!r}; "
            f"the trained model only knows {DEVICE_TYPES}"
        )

    return DeviceTarget(
        device_id=entry["device_id"],
        name=entry.get("name", entry["device_id"]),
        device_type=device_type,
        site=entry.get("site", "unknown"),
        host=entry["host"],
        interfaces=entry.get("interfaces", []),
        snmp_version=str(entry.get("snmp_version", "2c")),
        oid_profile=entry.get("oid_profile", "cisco"),
        community_env=entry.get("community_env", "AURA_SNMP_COMMUNITY"),
        oid_overrides=entry.get("oid_overrides", {}),
    )
