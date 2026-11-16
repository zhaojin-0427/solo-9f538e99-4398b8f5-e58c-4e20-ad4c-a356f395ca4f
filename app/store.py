"""SQLite metadata store, atomic version publishing and integrity checks.

Version publication follows this sequence, all while holding a SQLite
``BEGIN IMMEDIATE`` write transaction:

1. compare the caller's expected current version with the stored one;
2. allocate the next version number and render a complete GeoTIFF in a
   temporary file;
3. re-open and verify the rendered file, fsync it and atomically rename it to
   its immutable final path;
4. insert the version row and move the ``current_version`` pointer in the same
   transaction.

Consequently the current pointer only moves after a complete image exists on
disk.  A crash before commit leaves the previous version current and at worst
an unreferenced file, which startup reconciliation removes.  Versions are
never overwritten or deleted, so every published version stays downloadable.

Integrity verification ("quarantine") is deliberately separate from
publication.  A scan snapshots the registered versions and their expected size
and SHA-256, hashes the files without holding a database write lock, and then
applies the conclusions in one short write transaction.  Results are stored
per scan plus as the latest state of each version.  A version found missing or
corrupt stays queryable but is no longer downloadable until a later scan sees
the original bytes again.

The current pointer is only ever moved *down* by a scan: when the version that
was current at scan start is still current when results are applied and it
failed verification, the pointer moves to the highest-numbered older version
that passed this same scan (or to 0 when there is none).  A version published
while a scan is running therefore never gets rolled back, and a restored
version never promotes itself back to current.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import rasters
from .config import Settings
from .rasters import SourceBlock, iso_utc

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS versions (
    version        INTEGER PRIMARY KEY,
    day            TEXT    NOT NULL,
    status         TEXT    NOT NULL CHECK (status IN ('ready')),
    published_at   TEXT    NOT NULL,
    width          INTEGER NOT NULL,
    height         INTEGER NOT NULL,
    crs_wkt        TEXT    NOT NULL,
    crs_label      TEXT    NOT NULL,
    transform      TEXT    NOT NULL,
    bounds         TEXT    NOT NULL,
    res_x          REAL    NOT NULL,
    res_y          REAL    NOT NULL,
    dtype          TEXT    NOT NULL,
    nodata         INTEGER NOT NULL,
    sha256         TEXT    NOT NULL,
    size_bytes     INTEGER NOT NULL,
    block_meta     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS versions_day_idx ON versions(day);

CREATE TABLE IF NOT EXISTS state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS integrity_scans (
    scan_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at       TEXT    NOT NULL,
    finished_at      TEXT    NOT NULL,
    scanned_versions TEXT    NOT NULL,
    ok_count         INTEGER NOT NULL,
    missing_count    INTEGER NOT NULL,
    corrupt_count    INTEGER NOT NULL,
    current_before   INTEGER NOT NULL,
    current_after    INTEGER NOT NULL,
    fallback_applied INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS integrity_results (
    scan_id             INTEGER NOT NULL REFERENCES integrity_scans(scan_id),
    version             INTEGER NOT NULL,
    day                 TEXT    NOT NULL,
    outcome             TEXT    NOT NULL CHECK (outcome IN ('ok', 'missing', 'corrupt')),
    expected_sha256     TEXT    NOT NULL,
    expected_size_bytes INTEGER NOT NULL,
    actual_sha256       TEXT,
    actual_size_bytes   INTEGER,
    detail              TEXT,
    PRIMARY KEY (scan_id, version)
);

-- Latest verification conclusion for each version.  A row is only created by
-- a scan, so versions no scan has reached yet have no row (treated as
-- unverified and therefore still downloadable).
CREATE TABLE IF NOT EXISTS version_integrity (
    version    INTEGER PRIMARY KEY,
    outcome    TEXT    NOT NULL CHECK (outcome IN ('ok', 'missing', 'corrupt')),
    scan_id    INTEGER NOT NULL REFERENCES integrity_scans(scan_id),
    checked_at TEXT    NOT NULL,
    detail     TEXT
);
"""

CURRENT_STATE_KEY = "current_version"
ORPHAN_GRACE_SECONDS = 3600.0
HASH_CHUNK_BYTES = 1024 * 1024

# Serialises concurrent scan requests within this process.  SQLite still
# guards the actual writes; this only avoids two full scans racing and doing
# the same I/O twice.
_scan_lock = threading.Lock()

INTEGRITY_OK = "ok"
INTEGRITY_MISSING = "missing"
INTEGRITY_CORRUPT = "corrupt"
INTEGRITY_UNVERIFIED = "unverified"
INTEGRITY_OUTCOMES = (INTEGRITY_OK, INTEGRITY_MISSING, INTEGRITY_CORRUPT)


class VersionConflict(Exception):
    """The caller's expected version is not the published current version."""

    def __init__(self, expected_version: int, current_version: int, current: dict | None):
        super().__init__(
            f"expected version {expected_version} but current version is {current_version}"
        )
        self.expected_version = expected_version
        self.current_version = current_version
        self.current = current


class PublishError(Exception):
    """The new version could not be rendered or registered."""


class PublishBusy(Exception):
    """Another publication held the write lock for longer than the timeout."""


@dataclass(frozen=True)
class DownloadRecord:
    version: int
    day: str
    path: Path
    filename: str
    sha256: str
    size_bytes: int
    # Latest scan verdict; ``unverified`` means no scan has seen it yet.
    integrity_status: str = INTEGRITY_UNVERIFIED
    checked_at: str | None = None

    @property
    def downloadable(self) -> bool:
        return self.integrity_status in (INTEGRITY_OK, INTEGRITY_UNVERIFIED)


@dataclass(frozen=True)
class IntegrityRecord:
    """Expected bytes for one version, as registered at scan start."""

    version: int
    day: str
    path: Path
    sha256: str
    size_bytes: int


def connect(settings: Settings) -> sqlite3.Connection:
    conn = sqlite3.connect(
        str(settings.db_path),
        timeout=settings.sqlite_timeout_seconds,
        isolation_level=None,
    )
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {int(settings.sqlite_timeout_seconds * 1000)}")
    conn.execute("PRAGMA synchronous = FULL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_store(settings: Settings) -> None:
    """Create directories/schema, recover the pointer and clean leftovers."""

    settings.ensure_directories()
    conn = connect(settings)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)
        conn.execute(
            "INSERT OR IGNORE INTO state (key, value) VALUES (?, ?)",
            (CURRENT_STATE_KEY, "0"),
        )
        stored = _read_pointer(conn)
        max_version = _max_version(conn)
        if stored not in (0,) and _fetch_row(conn, stored) is None:
            log.warning("current version %s is missing; recovering to %s", stored, max_version)
            _write_pointer(conn, max_version)
        elif stored > max_version:
            log.warning("current version %s exceeds stored history; recovering to %s", stored, max_version)
            _write_pointer(conn, max_version)
    finally:
        conn.close()

    _reconcile_files(settings)


def _reconcile_files(settings: Settings) -> None:
    now = time.time()
    for path in settings.tmp_dir.glob("*"):
        try:
            if path.is_file() and now - path.stat().st_mtime > ORPHAN_GRACE_SECONDS:
                path.unlink()
        except OSError:
            log.warning("could not remove stale temporary file %s", path, exc_info=True)

    conn = connect(settings)
    try:
        registered = {int(row["version"]) for row in conn.execute("SELECT version FROM versions")}
    finally:
        conn.close()

    for path in settings.versions_dir.glob("v*.tif"):
        try:
            version = int(path.stem[1:])
        except ValueError:
            continue
        if version not in registered and now - path.stat().st_mtime > ORPHAN_GRACE_SECONDS:
            try:
                path.unlink()
                log.warning("removed orphaned mosaic file %s", path)
            except OSError:
                log.warning("could not remove orphaned mosaic %s", path, exc_info=True)


def _read_pointer(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT value FROM state WHERE key = ?", (CURRENT_STATE_KEY,)
    ).fetchone()
    if row is None:
        return 0
    try:
        return int(row["value"])
    except (TypeError, ValueError):
        return 0


def _write_pointer(conn: sqlite3.Connection, version: int) -> None:
    conn.execute(
        "INSERT INTO state (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (CURRENT_STATE_KEY, str(version)),
    )


def _max_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(version), 0) AS v FROM versions").fetchone()
    return int(row["v"])


def _fetch_row(conn: sqlite3.Connection, version: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM versions WHERE version = ?", (version,)).fetchone()


def _fetch_integrity_row(conn: sqlite3.Connection, version: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT outcome, scan_id, checked_at, detail FROM version_integrity WHERE version = ?",
        (version,),
    ).fetchone()


def _integrity_payload(row: sqlite3.Row | None) -> dict[str, Any]:
    """Describe the latest scan conclusion for a version.

    ``unverified`` means no completed scan has looked at this version yet; it
    stays downloadable so a freshly published version is never blocked merely
    because a scan has not run.
    """
    if row is None:
        return {
            "status": INTEGRITY_UNVERIFIED,
            "verified": False,
            "downloadable": True,
            "checked_at": None,
            "scan_id": None,
            "detail": None,
        }
    status = row["outcome"]
    return {
        "status": status,
        "verified": True,
        "downloadable": status == INTEGRITY_OK,
        "checked_at": row["checked_at"],
        "scan_id": int(row["scan_id"]),
        "detail": row["detail"],
    }


def _row_to_payload(
    row: sqlite3.Row, integrity_row: sqlite3.Row | None = None
) -> dict[str, Any]:
    integrity = _integrity_payload(integrity_row)
    return {
        "version": int(row["version"]),
        "day": row["day"],
        "status": row["status"],
        "published_at": row["published_at"],
        "downloadable": integrity["downloadable"],
        "integrity": integrity,
        "image": {
            "width": int(row["width"]),
            "height": int(row["height"]),
            "dtype": row["dtype"],
            "nodata": int(row["nodata"]),
            "crs": row["crs_label"],
            "crs_wkt": row["crs_wkt"],
            "transform": json.loads(row["transform"]),
            "bounds": json.loads(row["bounds"]),
            "res_x": float(row["res_x"]),
            "res_y": float(row["res_y"]),
            "sha256": row["sha256"],
            "size_bytes": int(row["size_bytes"]),
        },
        "blocks": json.loads(row["block_meta"]),
        "download_url": f"/api/v1/mosaics/{int(row['version'])}/download",
    }


def _current_payload(conn: sqlite3.Connection) -> tuple[int, dict[str, Any] | None]:
    version = _read_pointer(conn)
    if version <= 0:
        return 0, None
    row = _fetch_row(conn, version)
    if row is None:
        return 0, None
    return version, _row_to_payload(row, _fetch_integrity_row(conn, version))


def get_current_version(settings: Settings) -> int:
    conn = connect(settings)
    try:
        return _read_pointer(conn)
    finally:
        conn.close()


def get_current(settings: Settings) -> tuple[int, dict[str, Any] | None]:
    conn = connect(settings)
    try:
        return _current_payload(conn)
    finally:
        conn.close()


def get_version(settings: Settings, version: int) -> dict[str, Any] | None:
    conn = connect(settings)
    try:
        row = _fetch_row(conn, version)
        if row is None:
            return None
        return _row_to_payload(row, _fetch_integrity_row(conn, version))
    finally:
        conn.close()


def list_versions(
    settings: Settings, *, day: str | None = None, limit: int = 20, offset: int = 0
) -> tuple[list[dict[str, Any]], int]:
    conn = connect(settings)
    try:
        if day is None:
            total = conn.execute("SELECT COUNT(*) AS n FROM versions").fetchone()["n"]
            rows = conn.execute(
                "SELECT * FROM versions ORDER BY version DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        else:
            total = conn.execute(
                "SELECT COUNT(*) AS n FROM versions WHERE day = ?", (day,)
            ).fetchone()["n"]
            rows = conn.execute(
                "SELECT * FROM versions WHERE day = ? ORDER BY version DESC LIMIT ? OFFSET ?",
                (day, limit, offset),
            ).fetchall()
        return [
            _row_to_payload(row, _fetch_integrity_row(conn, int(row["version"])))
            for row in rows
        ], int(total)
    finally:
        conn.close()


def _download_record(
    settings: Settings, row: sqlite3.Row, integrity_row: sqlite3.Row | None
) -> DownloadRecord:
    version = int(row["version"])
    integrity = _integrity_payload(integrity_row)
    return DownloadRecord(
        version=version,
        day=row["day"],
        path=settings.versions_dir / f"v{version}.tif",
        filename=f"rainmosaic-{row['day']}-v{version}.tif",
        sha256=row["sha256"],
        size_bytes=int(row["size_bytes"]),
        integrity_status=integrity["status"],
        checked_at=integrity["checked_at"],
    )


def get_download(settings: Settings, version: int) -> DownloadRecord | None:
    conn = connect(settings)
    try:
        row = _fetch_row(conn, version)
        if row is None:
            return None
        return _download_record(settings, row, _fetch_integrity_row(conn, version))
    finally:
        conn.close()


def get_current_download(settings: Settings) -> DownloadRecord | None:
    conn = connect(settings)
    try:
        version, _ = _current_payload(conn)
        if version <= 0:
            return None
        row = _fetch_row(conn, version)
        if row is None:
            return None
        return _download_record(settings, row, _fetch_integrity_row(conn, version))
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Integrity verification
# --------------------------------------------------------------------------


def _snapshot_versions(
    settings: Settings, conn: sqlite3.Connection
) -> tuple[int, list[IntegrityRecord]]:
    """Registered versions at scan start and the current pointer then.

    Reading the pointer and the rows in one short read transaction means a
    concurrent publication either is fully visible (a new current version and
    a new row) or not visible at all.
    """

    conn.execute("BEGIN")
    try:
        current_version = _read_pointer(conn)
        rows = conn.execute(
            "SELECT version, day, sha256, size_bytes FROM versions ORDER BY version"
        ).fetchall()
    finally:
        _rollback(conn)
    return current_version, [
        IntegrityRecord(
            version=int(row["version"]),
            day=row["day"],
            path=settings.versions_dir / f"v{int(row['version'])}.tif",
            sha256=row["sha256"],
            size_bytes=int(row["size_bytes"]),
        )
        for row in rows
    ]


def _check_record(record: IntegrityRecord) -> dict[str, Any]:
    """Compare one file against its registered size and SHA-256."""

    result: dict[str, Any] = {
        "version": record.version,
        "day": record.day,
        "expected_sha256": record.sha256,
        "expected_size_bytes": record.size_bytes,
        "actual_sha256": None,
        "actual_size_bytes": None,
        "detail": None,
    }
    try:
        stat = record.path.stat()
    except FileNotFoundError:
        result["outcome"] = INTEGRITY_MISSING
        result["detail"] = "image file not found"
        return result
    except OSError as exc:
        result["outcome"] = INTEGRITY_CORRUPT
        result["detail"] = f"image file is unreadable: {exc}"
        return result

    if not record.path.is_file():
        result["outcome"] = INTEGRITY_MISSING
        result["detail"] = "image path is not a regular file"
        return result

    result["actual_size_bytes"] = int(stat.st_size)
    if stat.st_size != record.size_bytes:
        result["outcome"] = INTEGRITY_CORRUPT
        result["detail"] = (
            f"size mismatch: registered {record.size_bytes} bytes, "
            f"found {stat.st_size} bytes"
        )
        return result

    digest = hashlib.sha256()
    try:
        with record.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError as exc:
        result["outcome"] = INTEGRITY_CORRUPT
        result["detail"] = f"image file could not be read: {exc}"
        return result

    actual = digest.hexdigest()
    result["actual_sha256"] = actual
    if actual != record.sha256:
        result["outcome"] = INTEGRITY_CORRUPT
        result["detail"] = (
            f"SHA-256 mismatch: registered {record.sha256}, found {actual}"
        )
        return result

    result["outcome"] = INTEGRITY_OK
    return result


def run_integrity_scan(settings: Settings) -> dict[str, Any]:
    """Verify every version registered when the scan starts.

    Files are hashed without holding a write lock.  Conclusions are applied in
    one short write transaction that re-reads the current pointer, so a
    version published while the scan ran is never rolled back and a version
    that failed earlier but was not part of this scan is never selected as a
    fallback.
    """

    with _scan_lock:
        started = datetime.now(timezone.utc)
        conn = connect(settings)
        try:
            current_before, records = _snapshot_versions(settings, conn)
        finally:
            conn.close()

        results = [_check_record(record) for record in records]

        finished = datetime.now(timezone.utc)
        counts = {outcome: 0 for outcome in INTEGRITY_OUTCOMES}
        for result in results:
            counts[result["outcome"]] += 1

        conn = connect(settings)
        try:
            try:
                conn.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                    raise PublishBusy() from exc
                raise

            current_after = _read_pointer(conn)
            moved_to: int | None = None
            if (
                current_before > 0
                and current_before == current_after
                and any(
                    result["version"] == current_before
                    and result["outcome"] != INTEGRITY_OK
                    for result in results
                )
            ):
                # Only versions verified by *this* scan are eligible: a
                # version published after the snapshot is absent from
                # ``results`` and a version that failed an earlier scan has not
                # been re-verified here either.
                eligible = [
                    result["version"]
                    for result in results
                    if result["version"] < current_before
                    and result["outcome"] == INTEGRITY_OK
                ]
                moved_to = max(eligible) if eligible else 0
                _write_pointer(conn, moved_to)
                current_after = moved_to

            cursor = conn.execute(
                """
                INSERT INTO integrity_scans (
                    started_at, finished_at, scanned_versions, ok_count,
                    missing_count, corrupt_count, current_before, current_after,
                    fallback_applied
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    iso_utc(started),
                    iso_utc(finished),
                    json.dumps([result["version"] for result in results]),
                    counts[INTEGRITY_OK],
                    counts[INTEGRITY_MISSING],
                    counts[INTEGRITY_CORRUPT],
                    current_before,
                    current_after,
                    int(moved_to is not None),
                ),
            )
            scan_id = int(cursor.lastrowid)

            for result in results:
                conn.execute(
                    """
                    INSERT INTO integrity_results (
                        scan_id, version, day, outcome, expected_sha256,
                        expected_size_bytes, actual_sha256, actual_size_bytes,
                        detail
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        scan_id,
                        result["version"],
                        result["day"],
                        result["outcome"],
                        result["expected_sha256"],
                        result["expected_size_bytes"],
                        result["actual_sha256"],
                        result["actual_size_bytes"],
                        result["detail"],
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO version_integrity (
                        version, outcome, scan_id, checked_at, detail
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(version) DO UPDATE SET
                        outcome = excluded.outcome,
                        scan_id = excluded.scan_id,
                        checked_at = excluded.checked_at,
                        detail = excluded.detail
                    """,
                    (
                        result["version"],
                        result["outcome"],
                        scan_id,
                        iso_utc(finished),
                        result["detail"],
                    ),
                )
            conn.execute("COMMIT")
        except PublishBusy:
            _rollback(conn)
            raise
        except Exception:
            _rollback(conn)
            raise
        finally:
            conn.close()

        log.info(
            "integrity scan %s checked %s versions: %s ok, %s missing, %s corrupt; "
            "current %s -> %s",
            scan_id,
            len(results),
            counts[INTEGRITY_OK],
            counts[INTEGRITY_MISSING],
            counts[INTEGRITY_CORRUPT],
            current_before,
            current_after,
        )
        return _scan_payload(
            scan_id,
            started=started,
            finished=finished,
            counts=counts,
            results=results,
            current_before=current_before,
            current_after=current_after,
            fallback_applied=moved_to is not None,
        )


def _scan_payload(
    scan_id: int,
    *,
    started: datetime,
    finished: datetime,
    counts: dict[str, int],
    results: list[dict[str, Any]],
    current_before: int,
    current_after: int,
    fallback_applied: bool,
) -> dict[str, Any]:
    def view(result: dict[str, Any]) -> dict[str, Any]:
        return {
            "version": result["version"],
            "day": result["day"],
            "outcome": result["outcome"],
            "status": result["outcome"],
            "downloadable": result["outcome"] == INTEGRITY_OK,
            "expected_sha256": result["expected_sha256"],
            "expected_size_bytes": result["expected_size_bytes"],
            "actual_sha256": result["actual_sha256"],
            "actual_size_bytes": result["actual_size_bytes"],
            "detail": result["detail"],
        }

    ordered = sorted(results, key=lambda item: item["version"])
    return {
        "scan_id": scan_id,
        "started_at": iso_utc(started),
        "finished_at": iso_utc(finished),
        "scanned_versions": [result["version"] for result in ordered],
        "status": (
            "degraded"
            if counts[INTEGRITY_MISSING] or counts[INTEGRITY_CORRUPT]
            else INTEGRITY_OK
        ),
        "counts": {
            "total": len(results),
            "ok": counts[INTEGRITY_OK],
            "missing": counts[INTEGRITY_MISSING],
            "corrupt": counts[INTEGRITY_CORRUPT],
        },
        "current_before": current_before,
        "current_after": current_after,
        "current_version": current_after,
        "fallback_applied": fallback_applied,
        "ok": [view(result) for result in ordered if result["outcome"] == INTEGRITY_OK],
        "missing": [
            view(result) for result in ordered if result["outcome"] == INTEGRITY_MISSING
        ],
        "corrupt": [
            view(result) for result in ordered if result["outcome"] == INTEGRITY_CORRUPT
        ],
        "results": [view(result) for result in ordered],
    }


def get_integrity(settings: Settings, version: int) -> dict[str, Any] | None:
    """Latest integrity conclusion for a version, or ``None`` if unknown."""

    conn = connect(settings)
    try:
        if _fetch_row(conn, version) is None:
            return None
        return _integrity_payload(_fetch_integrity_row(conn, version))
    finally:
        conn.close()


def latest_scan(settings: Settings) -> dict[str, Any] | None:
    """Summary of the most recent completed scan, or ``None`` if none ran."""

    conn = connect(settings)
    try:
        row = conn.execute(
            "SELECT * FROM integrity_scans ORDER BY scan_id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return {
            "scan_id": int(row["scan_id"]),
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "scanned_versions": json.loads(row["scanned_versions"]),
            "status": (
                "degraded"
                if row["missing_count"] or row["corrupt_count"]
                else INTEGRITY_OK
            ),
            "counts": {
                "total": int(row["ok_count"] + row["missing_count"] + row["corrupt_count"]),
                "ok": int(row["ok_count"]),
                "missing": int(row["missing_count"]),
                "corrupt": int(row["corrupt_count"]),
            },
            "current_before": int(row["current_before"]),
            "current_after": int(row["current_after"]),
            "current_version": int(row["current_after"]),
            "fallback_applied": bool(row["fallback_applied"]),
        }
    finally:
        conn.close()


def publish(
    settings: Settings,
    *,
    day: str,
    expected_version: int,
    left: SourceBlock,
    right: SourceBlock,
) -> dict[str, Any]:
    """Render and publish a new version atomically."""

    temp_path = settings.tmp_dir / f"publish-{uuid.uuid4().hex}.tif"
    final_path: Path | None = None
    new_version: int | None = None
    conn = connect(settings)
    try:
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                raise PublishBusy() from exc
            raise
        current_version, current_payload = _current_payload(conn)
        if current_version != expected_version:
            raise VersionConflict(expected_version, current_version, current_payload)

        new_version = max(current_version, _max_version(conn)) + 1

        # Heavy work happens while the write lock is held.  Readers in WAL
        # mode keep seeing and downloading the old current version.
        info = rasters.write_mosaic(left, right, temp_path)
        rasters.verify_mosaic(temp_path, info)
        _fsync_file(temp_path)
        sha256 = rasters.sha256_file(temp_path)
        size_bytes = temp_path.stat().st_size

        final_path = settings.versions_dir / f"v{new_version}.tif"
        os.replace(temp_path, final_path)
        _fsync_dir(settings.versions_dir)

        published_at = iso_utc(datetime.now(timezone.utc))
        block_meta = [left.metadata(), right.metadata()]
        conn.execute(
            """
            INSERT INTO versions (
                version, day, status, published_at, width, height, crs_wkt,
                crs_label, transform, bounds, res_x, res_y, dtype, nodata,
                sha256, size_bytes, block_meta
            ) VALUES (?, ?, 'ready', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_version,
                day,
                published_at,
                info.width,
                info.height,
                info.crs.to_wkt(),
                info.crs.to_string() or info.crs.to_wkt(),
                json.dumps(list(info.transform)[:6]),
                json.dumps(list(info.bounds)),
                info.res_x,
                info.res_y,
                rasters.DTYPE,
                rasters.NODATA,
                sha256,
                size_bytes,
                json.dumps(block_meta),
            ),
        )
        _write_pointer(conn, new_version)
        row = _fetch_row(conn, new_version)
        assert row is not None
        conn.execute("COMMIT")
        log.info(
            "published version %s for %s (%sx%s, %s bytes)",
            new_version,
            day,
            info.width,
            info.height,
            size_bytes,
        )
        payload = _row_to_payload(row)
        return payload
    except VersionConflict:
        _rollback(conn)
        raise
    except PublishBusy:
        _rollback(conn)
        temp_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        _rollback(conn)
        temp_path.unlink(missing_ok=True)
        if final_path is not None and new_version is not None:
            # Only remove the renamed file if the version really is not in the
            # database (guards against an ambiguous commit outcome).
            try:
                row = _fetch_row(conn, new_version)
            except sqlite3.Error:
                row = None
            if row is None:
                final_path.unlink(missing_ok=True)
        log.exception("publishing mosaic failed")
        raise PublishError("mosaic generation or registration failed") from exc
    finally:
        conn.close()


def _rollback(conn: sqlite3.Connection) -> None:
    if conn.in_transaction:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            log.warning("rollback failed", exc_info=True)


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
