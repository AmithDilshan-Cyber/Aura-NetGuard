import pytest

from backend.app import db


@pytest.fixture(autouse=True)
def temp_database(tmp_path, monkeypatch):
    """Keep tests off the development database."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    yield
