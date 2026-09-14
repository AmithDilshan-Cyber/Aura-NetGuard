import json

import pytest

from backend.app.collectors.inventory import (
    STANDARD_OIDS,
    InventoryError,
    load_inventory,
)

VALID = {
    "devices": [
        {
            "device_id": "dev-001",
            "name": "core-rtr",
            "device_type": "core-router",
            "site": "Main",
            "host": "10.0.0.1",
            "interfaces": [1, 2],
        }
    ]
}


def _write(tmp_path, payload):
    path = tmp_path / "devices.json"
    path.write_text(json.dumps(payload))
    return path


def test_loads_a_valid_inventory(tmp_path):
    targets = load_inventory(_write(tmp_path, VALID))
    assert len(targets) == 1
    assert targets[0].device_id == "dev-001"
    assert targets[0].interfaces == [1, 2]


def test_missing_file_explains_how_to_fix_it(tmp_path):
    with pytest.raises(InventoryError, match="devices.example.json"):
        load_inventory(tmp_path / "nope.json")


def test_malformed_json_is_reported_clearly(tmp_path):
    path = tmp_path / "devices.json"
    path.write_text("{not json")
    with pytest.raises(InventoryError, match="not valid JSON"):
        load_inventory(path)


def test_empty_inventory_is_rejected(tmp_path):
    with pytest.raises(InventoryError, match="no devices"):
        load_inventory(_write(tmp_path, {"devices": []}))


@pytest.mark.parametrize("field", ["device_id", "host", "device_type"])
def test_missing_required_fields_are_rejected(tmp_path, field):
    entry = dict(VALID["devices"][0])
    del entry[field]
    with pytest.raises(InventoryError, match=field):
        load_inventory(_write(tmp_path, {"devices": [entry]}))


def test_unknown_device_type_is_rejected_with_the_valid_options(tmp_path):
    """The model only knows the five types it was trained on, so an
    unrecognised type must fail loudly rather than silently mispredict."""
    entry = {**VALID["devices"][0], "device_type": "toaster"}
    with pytest.raises(InventoryError, match="core-router"):
        load_inventory(_write(tmp_path, {"devices": [entry]}))


def test_duplicate_device_ids_are_rejected(tmp_path):
    entry = VALID["devices"][0]
    with pytest.raises(InventoryError, match="duplicate"):
        load_inventory(_write(tmp_path, {"devices": [entry, dict(entry)]}))


def test_bare_list_is_accepted_as_well_as_a_wrapped_object(tmp_path):
    assert len(load_inventory(_write(tmp_path, VALID["devices"]))) == 1


def test_community_comes_from_the_environment_not_the_file(tmp_path, monkeypatch):
    target = load_inventory(_write(tmp_path, VALID))[0]
    assert target.community() is None
    monkeypatch.setenv("AURA_SNMP_COMMUNITY", "s3cret")
    assert target.community() == "s3cret"


def test_oid_overrides_win_over_the_vendor_profile(tmp_path):
    entry = {**VALID["devices"][0], "oid_profile": "cisco",
             "oid_overrides": {"cpu_pct": "1.2.3.4"}}
    oids = load_inventory(_write(tmp_path, {"devices": [entry]}))[0].oids()
    assert oids["cpu_pct"] == "1.2.3.4"
    assert oids["ifInErrors"] == STANDARD_OIDS["ifInErrors"]
