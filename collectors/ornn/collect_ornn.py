#!/usr/bin/env python3
"""Private research collector for Ornn's public compute and token indices.

Adapted from a starter collector. Changes from the starter:
  - every response is archived BEFORE parsing, so a changed response shape
    never loses the raw bytes
  - failed requests and unparseable responses are recorded as fetch rows,
    so a gap in the data is always explained
  - one failure no longer aborts the whole run; the run exits non-zero instead

Dependencies: Python 3.10+ standard library only.
The raw response is archived byte for byte. SQLite is a convenience index.
Ornn's free API has not been verified end to end. Check the first runs by hand.
"""

from __future__ import annotations

import argparse
import os
import csv
import hashlib
import json
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

BASE = "https://api.ornnai.com"
DEFAULT_ROOT = Path(__file__).resolve().parent / "data"
SCHEMA = """
CREATE TABLE IF NOT EXISTS fetches (
  id INTEGER PRIMARY KEY,
  provider TEXT NOT NULL,
  kind TEXT NOT NULL,
  url TEXT NOT NULL,
  fetched_at_utc TEXT NOT NULL,
  raw_file TEXT,
  sha256 TEXT,
  observations_found INTEGER NOT NULL,
  status TEXT NOT NULL,
  problem TEXT
);
CREATE TABLE IF NOT EXISTS observations (
  id INTEGER PRIMARY KEY,
  provider TEXT NOT NULL,
  index_family TEXT NOT NULL,
  instrument TEXT NOT NULL,
  region TEXT NOT NULL DEFAULT '',
  observation_kind TEXT NOT NULL,
  source_time TEXT NOT NULL,
  value_text TEXT NOT NULL,
  unit TEXT NOT NULL,
  first_seen_at_utc TEXT NOT NULL,
  first_fetch_id INTEGER NOT NULL REFERENCES fetches(id),
  UNIQUE(provider, index_family, instrument, region, observation_kind, source_time, value_text)
);
CREATE INDEX IF NOT EXISTS idx_observation_time
ON observations(provider, index_family, instrument, source_time);
"""


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def normalize(kind: str, payload: dict, instrument: str | None = None) -> list[dict]:
    """Parse documented Ornn response shapes; keep vendor timestamps verbatim."""
    data = payload.get("data", [])
    if kind in ("gpu_types", "token_types"):
        return []
    if kind == "gpu_spot":
        if not isinstance(data, dict):
            raise ValueError("Expected an object for GPU current price")
        items = [data]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"Expected list for {kind}; got {type(data).__name__}")

    result = []
    for item in items:
        if kind in ("gpu_history", "gpu_daily", "gpu_spot"):
            gpu = item.get("gpu_type") or item.get("gpu_name") or instrument or payload.get("gpu_type")
            timestamp = (
                item.get("timestamp") if kind == "gpu_history" else
                item.get("date", payload.get("date")) if kind == "gpu_daily" else
                item.get("last_updated")
            )
            value = item.get("index_value")
            family = "OCPI"
            unit = "USD/GPU-hour"
            obs_kind = "current_price" if kind == "gpu_spot" else "daily_settlement"
            region = item.get("region") or ""
            name = gpu
        elif kind == "otpi":
            name = item.get("lab")
            timestamp = item.get("date", payload.get("date"))
            value = item.get("indexPerMtok")
            family = "OTPI"
            unit = "USD/million_tokens"
            obs_kind = "daily_settlement"
            region = ""
        else:
            raise ValueError(f"Unknown payload type: {kind}")

        if not name or not timestamp or value is None:
            raise ValueError(f"Missing required {kind} fields in: {item}")
        result.append({
            "provider": "ornn", "index_family": family, "instrument": str(name),
            "region": str(region), "observation_kind": obs_kind,
            "source_time": str(timestamp), "value_text": str(value), "unit": unit,
        })
    return result


class Collector:
    def __init__(self, root: Path):
        self.errors = 0
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.root / "indices.sqlite")
        self.conn.executescript(SCHEMA)

    def close(self):
        self.conn.close()

    def _archive(self, provider: str, label: str, content: bytes, url: str) -> tuple[str, str, str]:
        ts = now_utc()
        timestamp = ts.strftime("%Y%m%dT%H%M%S%fZ")
        folder = self.root / "raw" / provider / ts.strftime("%Y/%m/%d")
        folder.mkdir(parents=True, exist_ok=True)
        raw_path = folder / f"{timestamp}_{label}.json"
        raw_path.write_bytes(content)  # exact API response bytes
        meta = {
            "provider": provider, "collected_at_utc": ts.isoformat(),
            "source_url": url, "sha256": hashlib.sha256(content).hexdigest(),
            "raw_file": str(raw_path.relative_to(self.root)),
        }
        (folder / f"{timestamp}_{label}.meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        return ts.isoformat(), str(raw_path.relative_to(self.root)), meta["sha256"]

    def _record_fetch(self, kind, url, fetched_at, raw_rel, sha256, n_rows, status, problem):
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO fetches(provider,kind,url,fetched_at_utc,raw_file,sha256,"
                "observations_found,status,problem) VALUES (?,?,?,?,?,?,?,?,?)",
                ("ornn", kind, url, fetched_at, raw_rel, sha256, n_rows, status, problem),
            )
        return cursor.lastrowid

    def retrieve(self, kind: str, path: str, params: dict | None = None, instrument: str | None = None):
        """Fetch one endpoint. Archive first, parse second. Returns the payload, or None on failure."""
        qs = "?" + urlencode(params) if params else ""
        url = f"{BASE}{path}{qs}"
        label = kind + ("_" + quote(instrument, safe="").replace("%", "_") if instrument else "")
        request = Request(url, headers={
            "User-Agent": "gpu-price-mix-research/0.1 (private academic research)",
            "Accept": "application/json",
        })

        # 1. Fetch, with limited retries. A final failure is recorded, not raised.
        content = None
        failure = None
        for attempt in range(3):
            try:
                with urlopen(request, timeout=25) as response:
                    content = response.read()
                failure = None
                break
            except HTTPError as error:
                failure = f"http_{error.code}"
                body = error.read()
                if body:
                    content = body       # keep the error body as evidence
                if error.code not in (429, 500, 502, 503, 504):
                    break
            except (URLError, TimeoutError) as error:
                failure = f"network: {error}"
            if attempt < 2:
                time.sleep(2 ** attempt + 1)

        # 2. Archive whatever came back, before trying to understand it
        fetched_at = now_utc().isoformat()
        raw_rel = None
        sha256 = None
        if content is not None:
            fetched_at, raw_rel, sha256 = self._archive("ornn", label, content, url)

        if failure is not None:
            self._record_fetch(kind, url, fetched_at, raw_rel, sha256, 0, "failed", failure[:300])
            self.errors += 1
            print(f"{kind:14} {instrument or 'all':14} FAILED {failure}", file=sys.stderr)
            return None

        # 3. Parse. A changed response shape is recorded with the raw bytes already saved.
        try:
            payload = json.loads(content)
            if not isinstance(payload, dict) or payload.get("success") is False:
                raise ValueError(f"unexpected response: {str(payload)[:200]}")
            rows = normalize(kind, payload, instrument)
        except (ValueError, json.JSONDecodeError) as error:
            self._record_fetch(kind, url, fetched_at, raw_rel, sha256, 0, "parse_error", str(error)[:300])
            self.errors += 1
            print(f"{kind:14} {instrument or 'all':14} PARSE ERROR {error}", file=sys.stderr)
            return None

        # 4. Store parsed observations. Identical prints are ignored, revisions are kept.
        fetch_id = self._record_fetch(kind, url, fetched_at, raw_rel, sha256, len(rows), "ok", None)
        with self.conn:
            for row in rows:
                self.conn.execute(
                    """INSERT OR IGNORE INTO observations(
                    provider,index_family,instrument,region,observation_kind,
                    source_time,value_text,unit,first_seen_at_utc,first_fetch_id
                    ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (row["provider"], row["index_family"], row["instrument"],
                     row["region"], row["observation_kind"], row["source_time"],
                     row["value_text"], row["unit"], fetched_at, fetch_id),
                )
        print(f"{kind:14} {instrument or 'all':14} {len(rows):5} parsed | {raw_rel}")
        return payload

    def _free_gpu_names(self):
        catalog = self.retrieve("gpu_types", "/api/gpu-types-free")
        if catalog is None:
            raise RuntimeError("Could not read Ornn's free GPU list")
        names = set()
        for item in catalog.get("data", []):
            if "gpu_name" in item:
                names.add(item["gpu_name"])
        if not names:
            raise RuntimeError("No public GPUs returned by Ornn API")
        return sorted(names)

    def backfill(self):
        for name in self._free_gpu_names():
            encoded = quote(name, safe="")
            self.retrieve("gpu_history", f"/api/gpu/{encoded}/index-history", instrument=name)
            time.sleep(0.35)
        self.retrieve("token_types", "/api/token-types-free")
        # Ornn documentation has varying descriptions of the token free-tier window.
        # Ask for 90 days and let the service enforce the authorized limit.
        today = now_utc().date()
        self.retrieve("otpi", "/api/otpi", {
            "startDate": (today - timedelta(days=89)).isoformat(),
            "endDate": today.isoformat(),
        })

    def daily(self):
        self.retrieve("gpu_daily", "/api/daily-index/all")
        self.retrieve("otpi", "/api/otpi")

    def intraday(self):
        for name in self._free_gpu_names():
            encoded = quote(name, safe="")
            self.retrieve("gpu_spot", f"/api/gpu/{encoded}", instrument=name)
            time.sleep(0.35)

    def import_silicon(self, source: Path):
        """Stage a legitimately obtained vendor CSV or JSON unchanged, without guessing columns."""
        if source.suffix.lower() not in (".csv", ".json") or not source.is_file():
            raise ValueError("Expected an existing .csv or .json export")
        ts = now_utc()
        folder = self.root / "raw" / "silicon_data" / ts.strftime("%Y/%m/%d")
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / f"{ts.strftime('%Y%m%dT%H%M%S%fZ')}_{source.name}"
        shutil.copyfile(source, dest)
        meta = {
            "provider": "silicon_data", "collected_at_utc": ts.isoformat(),
            "source_type": "licensed_manual_import", "original_filename": source.name,
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            "note": "No automatic normalization: vendor schema and entitlement must be confirmed",
        }
        (folder / (dest.name + ".meta.json")).write_text(json.dumps(meta, indent=2) + "\n")
        print(f"Imported vendor export (unaltered): {dest}")

    def stats(self):
        print("\nObservation counts by provider / family / kind:")
        for row in self.conn.execute("""SELECT provider,index_family,observation_kind,
            COUNT(*),MIN(source_time),MAX(source_time)
            FROM observations GROUP BY 1,2,3 ORDER BY 1,2,3"""):
            print("  ", row)
        print("\nFetches by kind / status:")
        for row in self.conn.execute(
            "SELECT kind,status,COUNT(*) FROM fetches GROUP BY 1,2 ORDER BY 1,2"
        ):
            print("  ", row)

    def export_csv(self):
        target = self.root / "derived" / "observations.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        query = """SELECT provider,index_family,instrument,region,observation_kind,
            source_time,value_text,unit,first_seen_at_utc FROM observations
            ORDER BY index_family,instrument,source_time"""
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["provider", "index_family", "instrument", "region",
                             "observation_kind", "source_time", "value", "unit", "first_seen_at_utc"])
            writer.writerows(self.conn.execute(query))
        print(f"Exported {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("backfill", "daily", "intraday", "stats", "export", "import-silicon"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Private data directory (default: ./data)")
    parser.add_argument("--file", type=Path, help="Licensed Silicon Data CSV/JSON for import-silicon")
    args = parser.parse_args()
    collector = Collector(args.root)
    try:
        if args.command == "backfill": collector.backfill()
        elif args.command == "daily": collector.daily()
        elif args.command == "intraday": collector.intraday()
        elif args.command == "stats": collector.stats()
        elif args.command == "export": collector.export_csv()
        elif args.command == "import-silicon":
            if not args.file:
                parser.error("import-silicon requires --file path/to/export.csv")
            collector.import_silicon(args.file)
    finally:
        errors = collector.errors
        collector.close()
    ping_healthcheck(errors == 0)
    if errors:
        raise SystemExit(f"{errors} fetch(es) failed or could not be parsed; see the fetches table")


def ping_healthcheck(all_ok):
    """Tell an external monitor the run happened. Uses HEALTHCHECK_URL_ORNN if set."""
    url = os.environ.get("HEALTHCHECK_URL_ORNN")
    if not url:
        return
    if not all_ok:
        url = url.rstrip("/") + "/fail"
    try:
        urlopen(url, timeout=10)
    except OSError as error:
        print(f"healthcheck ping failed: {error}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, HTTPError, URLError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
