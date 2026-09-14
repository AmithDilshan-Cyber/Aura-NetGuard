# Connecting Aura-NetGuard to a real network

The simulator exists so the system can be developed and demonstrated without
hardware. This guide covers replacing it with real telemetry.

**Read §4 before planning any research around this.** The collector is the
easy part; getting training labels is the part that decides whether
prediction is possible at all.

---

## 1. The integration point

Exactly one interface separates "where telemetry comes from" from everything
else (`backend/app/collectors/base.py`):

```python
class Collector(Protocol):
    devices: list[Device]
    def step(self) -> list[MetricSnapshot]: ...
```

`FleetSimulator` and `SNMPCollector` both satisfy it. The feature pipeline,
model, SHAP explainer, alert engine and dashboard are all downstream and need
no changes when you swap collectors.

Select one at runtime:

```bash
export AURA_COLLECTOR=simulator     # default, no hardware
export AURA_COLLECTOR=snmp          # poll real devices
```

## 2. Setting up SNMP collection

```bash
pip install -r requirements-snmp.txt

cp config/devices.example.json config/devices.json
$EDITOR config/devices.json

export AURA_SNMP_COMMUNITY='your-read-only-community'
export AURA_COLLECTOR=snmp
export AURA_INVENTORY=config/devices.json

uvicorn backend.app.main:app
```

Credentials are never stored in the inventory file — each device names an
environment variable instead, so the file stays safe to commit.

Find your interface indexes with:

```bash
snmpwalk -v2c -c <community> <host> 1.3.6.1.2.1.2.2.1.2
```

Ask your network administrator for a **read-only** community string. This
system only ever reads; it never writes device configuration.

### Which OIDs

`IF-MIB` counters (errors, discards, octets, speed) are standard across
vendors and already configured. CPU, memory and temperature are
vendor-specific — there is no universal OID. Cisco defaults ship in
`VENDOR_OID_PROFILES`; for anything else, set `oid_overrides` per device
(the example file shows MikroTik values).

Latency, jitter and packet loss are **not available over SNMP at all** and
are measured by active ICMP probing (`collectors/icmp.py`). In a serious
deployment, drive these from a probe agent near your users, or read Cisco IP
SLA / Juniper RPM results, so you measure the path users' traffic actually
takes rather than the path from the monitoring host.

## 3. Why the counter handling matters

`ifInErrors` and friends are monotonically increasing totals, not rates.
Subtracting consecutive polls naively causes two classic false-alert bugs,
both handled in `collectors/counters.py`:

| Event | Naive result | What this code does |
|---|---|---|
| Counter32 wraps at 2³² | Negative delta, or a huge fake spike | Accounts for the rollover |
| Device reboots (counter → 0) | Identical-looking huge fake spike | Detects it and **discards** the interval |

Telling those apart is the whole trick. Two signals are used: `sysUpTime`
going backwards proves a reboot, and — when uptime is unavailable — a
"wrap" implying a physically impossible rate must have been a reset.

When no trustworthy rate can be derived, the collector emits nothing for
that device this cycle. A gap in the data is recoverable; a fabricated spike
pages someone at 3am for nothing.

One deliberate exception: a device that is **completely unreachable** is
always reported, even though ICMP then yields no latency figure. Dropping it
would hide the single most important event the system exists to surface.

## 4. The hard part: labels

The model is trained to answer *"will this device fail within the next ~7.5
minutes?"*. Learning that requires historical examples of devices failing —
timestamped. A real collector cannot supply them: `failure_event` is always
`False` on real data, because ground truth simply is not available at poll
time.

Labels have to come from elsewhere:

- **Syslog** — link-down / interface-down events, reload messages
- **Ticketing history** — incident records with timestamps
- **Up/down history** — from your existing monitoring (Nagios, Zabbix, PRTG, LibreNMS)

Then align each failure timestamp with the telemetry window preceding it, in
place of the labelling loop in `backend/app/ml/dataset.py`.

### The realistic obstacle

A healthy campus network may produce only a handful of genuine device
failures in several months. That is excellent for the network and a serious
problem for supervised learning — there may not be enough positive examples
to train on at all.

Options if that happens, in rough order of practicality:

1. **Collect for longer** before attempting prediction. Run in collect-only
   mode (below) for weeks or months first.
2. **Widen the definition of "failure"** beyond hard outages: interface
   flaps, threshold breaches, degradation episodes that required
   intervention. More events, still operationally meaningful.
3. **Switch to unsupervised anomaly detection** (isolation forest,
   autoencoder) which needs no labels — at the cost of losing the lead-time
   prediction framing that makes this project distinctive.
4. **Keep the synthetic evaluation** and present real data as a deployment
   feasibility study. For an undergraduate project this is a perfectly
   defensible position, provided you state it plainly.

### Collect-only mode

You do not need a trained model to start gathering data. If no model file is
present, the app collects and stores telemetry and simply skips prediction:

```bash
rm -f backend/app/ml/model_store/failure_predictor.joblib
AURA_COLLECTOR=snmp uvicorn backend.app.main:app
```

Telemetry accumulates in SQLite (`backend/data/aura_netguard.db`). Note that
`METRIC_RETENTION_HOURS` in `backend/app/db.py` prunes history after 6 hours
— **raise it substantially before a long collection run**, or export
elsewhere.

## 5. Do not expect the shipped model to work on real data

The bundled model learned the simulator's scales and dynamics. Pointing it at
real telemetry will produce meaningless predictions. It must be retrained on
real, labelled data — which is what §4 is about.

Also check these assumptions still hold for your deployment:

| Assumption | Where | Note |
|---|---|---|
| 30-second poll interval | `simulator.STEP_SECONDS` | Polling every 5 min changes what the horizon means |
| 10-step (5 min) feature window | `features.WINDOW_STEPS` | |
| 15-step (~7.5 min) horizon | `features.PREDICTION_HORIZON_STEPS` | |
| Five device types | `simulator.DEVICE_TYPES` | The inventory rejects unknown types |
| Metric critical thresholds | `alerts/explain.py` | Tune to your environment |

Polling hundreds of devices every 30 seconds is also a real load — both on
the monitoring host and the devices. Start with a handful and measure.

## 6. Status and testing

| Component | Tested |
|---|---|
| Counter rate conversion, wrap/reset detection | Yes — `test_counters.py` |
| ICMP output parsing (Linux + BSD formats) | Yes — `test_icmp.py` |
| Inventory loading and validation | Yes — `test_inventory.py` |
| Collector aggregation and snapshot assembly | Yes — `test_snmp_collector.py`, via a fake transport |
| **pysnmp transport against real hardware** | **No** |

The SNMP transport is isolated behind the `SnmpSession` protocol precisely
so everything above it could be tested without hardware. Verify it against a
single device before pointing it at a fleet — and if you prefer, implement
`SnmpSession` over easysnmp, a Prometheus scrape, or your existing
monitoring system's API instead.
