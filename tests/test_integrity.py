"""Integrity scans: verdicts, quarantine, fallback and scan-time races."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app import store
from tests.conftest import make_geotiff, submit_payload

A_TILE = np.array([[1, 2], [3, 4]], dtype=np.int16)
B_TILE = np.array([[7, 8], [9, 10]], dtype=np.int16)


def _submit(client, *, expected_version: int, value: int | None = None):
    a = A_TILE if value is None else np.full((2, 2), value, dtype=np.int16)
    b = B_TILE if value is None else np.full((2, 2), value + 100, dtype=np.int16)
    data, files = submit_payload(
        block_a=make_geotiff(a),
        block_b=make_geotiff(b, x0=0.01),
        expected_version=expected_version,
    )
    return client.post("/api/v1/mosaics", data=data, files=files)


def _scan(client):
    response = client.post("/api/v1/integrity/scans")
    assert response.status_code == 200, response.text
    return response.json()


def _outcomes(body) -> dict[int, str]:
    return {item["version"]: item["outcome"] for item in body["results"]}


def _publish_three(client) -> None:
    assert _submit(client, expected_version=0, value=1).status_code == 201
    assert _submit(client, expected_version=1, value=2).status_code == 201
    assert _submit(client, expected_version=2, value=3).status_code == 201


def test_clean_scan_reports_ok_and_keeps_current(client):
    _publish_three(client)
    body = _scan(client)
    assert body["counts"] == {"total": 3, "ok": 3, "missing": 0, "corrupt": 0}
    assert body["status"] == "ok"
    assert _outcomes(body) == {1: "ok", 2: "ok", 3: "ok"}
    assert body["current_before"] == 3 and body["current_after"] == 3
    assert body["fallback_applied"] is False
    assert body["scanned_versions"] == [1, 2, 3]
    assert client.get("/api/v1/mosaics/current").json()["current_version"] == 3
    assert client.get("/api/v1/mosaics/3/download").status_code == 200


def test_corrupt_file_is_quarantined_but_metadata_stays_visible(client, settings):
    _publish_three(client)
    path = settings.versions_dir / "v2.tif"
    original = path.read_bytes()
    path.write_bytes(original + b"tampered")

    body = _scan(client)
    assert _outcomes(body) == {1: "ok", 2: "corrupt", 3: "ok"}
    assert body["counts"]["corrupt"] == 1
    assert body["status"] == "degraded"
    corrupt = body["corrupt"][0]
    assert corrupt["version"] == 2
    assert corrupt["actual_sha256"] != corrupt["expected_sha256"]
    assert corrupt["detail"]

    # Metadata remains queryable, including the integrity verdict.
    meta = client.get("/api/v1/mosaics/2")
    assert meta.status_code == 200
    payload = meta.json()["version"]
    assert payload["downloadable"] is False
    assert payload["integrity"]["status"] == "corrupt"

    listed = client.get("/api/v1/mosaics").json()["items"]
    assert {item["version"]: item["downloadable"] for item in listed} == {
        1: True,
        2: False,
        3: True,
    }

    # Downloading the quarantined version is refused.
    download = client.get("/api/v1/mosaics/2/download")
    assert download.status_code == 410
    assert download.json()["detail"]["code"] == "image_unavailable"
    assert client.get("/api/v1/mosaics/1/download").status_code == 200
    assert client.get("/api/v1/mosaics/3/download").status_code == 200


def test_missing_file_is_quarantined(client, settings):
    assert _submit(client, expected_version=0).status_code == 201
    (settings.versions_dir / "v1.tif").unlink()

    body = _scan(client)
    assert _outcomes(body) == {1: "missing"}
    assert body["missing"][0]["actual_sha256"] is None
    assert body["missing"][0]["actual_size_bytes"] is None
    assert body["status"] == "degraded"

    assert client.get("/api/v1/mosaics/1").status_code == 200
    download = client.get("/api/v1/mosaics/1/download")
    assert download.status_code == 410
    assert download.json()["detail"]["code"] == "image_unavailable"


def test_current_corrupt_falls_back_to_newest_verified_older_version(client, settings):
    _publish_three(client)
    (settings.versions_dir / "v3.tif").write_bytes(b"not the registered bytes")

    body = _scan(client)
    assert _outcomes(body) == {1: "ok", 2: "ok", 3: "corrupt"}
    assert body["current_before"] == 3
    assert body["current_after"] == 2
    assert body["fallback_applied"] is True

    current = client.get("/api/v1/mosaics/current").json()
    assert current["current_version"] == 2
    assert client.get("/api/v1/download/current").status_code == 200
    assert client.get("/api/v1/mosaics/3/download").status_code == 410


def test_current_corrupt_with_no_verified_older_version_clears_pointer(client, settings):
    _publish_three(client)
    (settings.versions_dir / "v3.tif").write_bytes(b"bad")
    (settings.versions_dir / "v2.tif").unlink()
    (settings.versions_dir / "v1.tif").write_bytes(b"also bad")

    body = _scan(client)
    assert _outcomes(body) == {1: "corrupt", 2: "missing", 3: "corrupt"}
    assert body["current_after"] == 0
    assert body["fallback_applied"] is True
    assert client.get("/api/v1/mosaics/current").status_code == 404
    assert client.get("/api/v1/download/current").status_code == 404
    # All metadata is still readable.
    for version in (1, 2, 3):
        assert client.get(f"/api/v1/mosaics/{version}").status_code == 200


def test_fallback_skips_versions_that_failed_this_scan(client, settings):
    """Fallback only considers versions this scan verified ok; a version that
    is still corrupt must never be selected even if it is newer."""

    _publish_three(client)
    (settings.versions_dir / "v3.tif").write_bytes(b"bad")
    assert _scan(client)["current_after"] == 2

    # Now the current version 2 is also damaged; the newest version this scan
    # verified ok is 1, so the pointer goes to 1 and not to v3.
    (settings.versions_dir / "v2.tif").write_bytes(b"bad")
    body = _scan(client)
    assert _outcomes(body) == {1: "ok", 2: "corrupt", 3: "corrupt"}
    assert body["current_before"] == 2
    assert body["current_after"] == 1
    assert body["fallback_applied"] is True
    assert client.get("/api/v1/mosaics/1/download").status_code == 200
    assert client.get("/api/v1/mosaics/3/download").status_code == 410


def test_restoring_original_bytes_clears_quarantine_without_usurping_current(
    client, settings
):
    _publish_three(client)
    path = settings.versions_dir / "v2.tif"
    original = path.read_bytes()
    path.write_bytes(b"corrupt")

    assert _scan(client)["current_after"] == 3
    assert _outcomes(_scan(client))[2] == "corrupt"

    # Restore the exact bytes; the next scan clears the quarantine.
    path.write_bytes(original)
    body = _scan(client)
    assert _outcomes(body)[2] == "ok"
    assert body["current_after"] == 3
    assert body["fallback_applied"] is False
    assert client.get("/api/v1/mosaics/2/download").status_code == 200
    assert client.get("/api/v1/mosaics/current").json()["current_version"] == 3


def test_restored_version_does_not_reclaim_earlier_fallback(client, settings):
    """Version 3 current -> corrupt -> fallback to 2; restoring 3 keeps 2
    current because a repaired version must not promote itself."""

    _publish_three(client)
    path = settings.versions_dir / "v3.tif"
    original = path.read_bytes()
    path.write_bytes(b"bad")
    assert _scan(client)["current_after"] == 2

    path.write_bytes(original)
    body = _scan(client)
    assert _outcomes(body)[3] == "ok"
    assert body["current_after"] == 2
    assert client.get("/api/v1/mosaics/current").json()["current_version"] == 2


def test_verdicts_and_scan_summary_are_persisted(client):
    _publish_three(client)
    body = _scan(client)
    assert client.get("/api/v1/integrity/scans/latest").json() == {
        "scan_id": body["scan_id"],
        "started_at": body["started_at"],
        "finished_at": body["finished_at"],
        "scanned_versions": [1, 2, 3],
        "status": "ok",
        "counts": {"total": 3, "ok": 3, "missing": 0, "corrupt": 0},
        "current_before": 3,
        "current_after": 3,
        "current_version": 3,
        "fallback_applied": False,
    }
    verdict = client.get("/api/v1/mosaics/1/integrity").json()
    assert verdict["version"] == 1
    assert verdict["status"] == "ok"

    # A second scan supersedes the first conclusion for each version.
    (client.app.state.settings.versions_dir / "v1.tif").write_bytes(b"bad")
    second = _scan(client)
    assert second["scan_id"] == body["scan_id"] + 1
    assert client.get("/api/v1/mosaics/1/integrity").json()["status"] == "corrupt"
    assert client.get("/api/v1/integrity/scans/latest").json()["scan_id"] == second["scan_id"]


def test_no_scan_yet_is_reported(client):
    assert client.get("/api/v1/integrity/scans/latest").status_code == 404
    assert _submit(client, expected_version=0).status_code == 201
    verdict = client.get("/api/v1/mosaics/1/integrity").json()
    assert verdict["status"] == "unverified"
    assert verdict["downloadable"] is True
    assert client.get("/api/v1/mosaics/99/integrity").status_code == 404
    # Unscanned versions stay downloadable (the pre-scan behaviour).
    assert client.get("/api/v1/mosaics/1/download").status_code == 200


def test_scan_ignores_versions_published_while_it_runs(client, settings):
    """A scan snapshots at start; a late result must not roll back a version
    published during the scan, nor pick the unchecked version as fallback."""

    _publish_three(client)
    (settings.versions_dir / "v3.tif").write_bytes(b"bad")

    started = threading.Event()
    release = threading.Event()
    real_check = store._check_record

    def slow_check(record):
        if record.version == 1:
            started.set()
            assert release.wait(timeout=30), "scan never released"
        return real_check(record)

    store._check_record = slow_check
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(lambda: client.post("/api/v1/integrity/scans"))
            assert started.wait(timeout=10), "scan did not start"

            # Publish version 4 while the scan is still hashing.
            published = _submit(client, expected_version=3, value=42)
            assert published.status_code == 201, published.text
            assert client.get("/api/v1/mosaics/current").json()["current_version"] == 4

            release.set()
            body = pending.result(timeout=60).json()
    finally:
        store._check_record = real_check

    assert body["scanned_versions"] == [1, 2, 3]
    assert body["current_before"] == 3
    # Version 3 failed, but it is no longer current; version 4 (unchecked by
    # this scan) is current and must not be replaced by the fallback.
    assert body["current_after"] == 4
    assert body["fallback_applied"] is False
    assert client.get("/api/v1/mosaics/current").json()["current_version"] == 4
    # The late v3 verdict is still recorded, so v3 is quarantined.
    assert client.get("/api/v1/mosaics/3/download").status_code == 410
    assert client.get("/api/v1/mosaics/4/download").status_code == 200
    assert client.get("/api/v1/mosaics/4/integrity").json()["status"] == "unverified"


def test_late_scan_never_selects_unverified_version_as_fallback(client, settings):
    """Current is 1 and fails; version 2 was published mid-scan and so was
    not checked by this scan. The pointer must clear, not jump to 2."""

    assert _submit(client, expected_version=0).status_code == 201
    (settings.versions_dir / "v1.tif").write_bytes(b"bad")

    started = threading.Event()
    release = threading.Event()
    real_check = store._check_record

    def slow_check(record):
        started.set()
        assert release.wait(timeout=30), "scan never released"
        return real_check(record)

    store._check_record = slow_check
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(lambda: client.post("/api/v1/integrity/scans"))
            assert started.wait(timeout=10)
            assert _submit(client, expected_version=1, value=5).status_code == 201
            release.set()
            body = pending.result(timeout=60).json()
    finally:
        store._check_record = real_check

    assert body["scanned_versions"] == [1]
    assert body["current_before"] == 1
    # Version 2 became current during the scan; the scan reports the real
    # pointer and does not move it, because the version it condemned is no
    # longer current and v2 was never checked.
    assert body["current_after"] == 2
    assert body["fallback_applied"] is False
    assert client.get("/api/v1/mosaics/current").json()["current_version"] == 2


def test_scan_of_empty_store_succeeds(client):
    body = _scan(client)
    assert body["counts"] == {"total": 0, "ok": 0, "missing": 0, "corrupt": 0}
    assert body["current_after"] == 0
    assert body["scanned_versions"] == []
    assert body["results"] == []


def test_verdict_survives_restart(restartable_client):
    first = restartable_client()
    _publish_three(first)
    (first.app.state.settings.versions_dir / "v2.tif").write_bytes(b"bad")
    assert _scan(first)["current_after"] == 3
    first.__exit__(None, None, None)

    restarted = restartable_client()
    assert restarted.get("/api/v1/mosaics/2/download").status_code == 410
    assert restarted.get("/api/v1/mosaics/2").json()["version"]["integrity"]["status"] == "corrupt"
    latest = restarted.get("/api/v1/integrity/scans/latest").json()
    assert latest["scanned_versions"] == [1, 2, 3]


def test_size_only_change_is_corrupt(client, settings):
    """The check is on the registered bytes, not just parseability: a
    same-length but different-byte file must be rejected too."""

    assert _submit(client, expected_version=0).status_code == 201
    path = settings.versions_dir / "v1.tif"
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))

    body = _scan(client)
    assert _outcomes(body) == {1: "corrupt"}
    assert body["corrupt"][0]["actual_size_bytes"] == len(data)
    assert client.get("/api/v1/mosaics/1/download").status_code == 410
