"""Integration health check for the docker-compose Qdrant service."""

import time

import pytest


@pytest.mark.integration
def test_qdrant_health_returns_server_version() -> None:
    from qdrant_client import QdrantClient

    client = QdrantClient(url="http://localhost:6333", timeout=5)
    # `make up` returns before Qdrant binds its port; poll briefly instead of racing it.
    deadline = time.monotonic() + 30.0
    while True:
        try:
            info = client.info()
            break
        except Exception:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.5)

    assert info.version, "Qdrant returned an empty version"
    assert info.version[0].isdigit()
