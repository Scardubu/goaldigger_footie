"""Standalone freshness monitor.

Runs freshness validations across key tables and emits a JSON artifact plus
stdout summary exit code signalling.

Usage (examples):
  python ingestion/freshness_monitor.py --tables fixtures:last_updated:120 results:updated_at:180
  python ingestion/freshness_monitor.py --default-age 1440

Arguments:
  --tables  space separated entries table:timestamp_column:max_age_minutes
  --default-age  fallback max age (minutes) if not specified per table
  --db-path override database path (defaults to DATA_DB_PATH env or data/football.db)
  --artifact-dir override output directory (defaults to data/freshness_runs)
  --fail-on-warn treat any failing freshness check as non-zero exit code

Artifact file schema:
{
  "generated_at": iso8601,
  "db_path": str,
  "tables": [ { table, ts_column, max_age_minutes, latest, age_minutes, ok, error? } ],
  "overall_ok": bool
}
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from typing import Dict, List

DEFAULT_DB = os.getenv("DATA_DB_PATH", os.path.join("data", "football.db"))
ARTIFACT_DIR = os.path.join("data", "freshness_runs")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Reuse parsing heuristics similar to pipeline controller

def parse_timestamp(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value > 1e12:
            return value / 1000.0
        if value > 1e10:
            return value / 1000.0
        return float(value)
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return time.mktime(time.strptime(value[:19], fmt))
            except Exception:  # noqa
                continue
    return None

def check_table(conn, table: str, ts_col: str, max_age_min: int) -> Dict:
    out = {
        "table": table,
        "ts_column": ts_col,
        "max_age_minutes": max_age_min,
        "latest": None,
        "age_minutes": None,
        "ok": False,
    }
    try:
        cur = conn.execute(
            f"SELECT {ts_col} FROM {table} WHERE {ts_col} IS NOT NULL ORDER BY {ts_col} DESC LIMIT 1"
        )
        row = cur.fetchone()
        if not row:
            out["error"] = "no_rows_with_timestamp"
            return out
        latest = row[0]
        out["latest"] = latest
        ts_val = parse_timestamp(latest)
        if ts_val is None:
            out["error"] = "unparseable_timestamp"
            return out
        age = (time.time() - ts_val) / 60.0
        out["age_minutes"] = round(age, 2)
        out["ok"] = age <= max_age_min
        return out
    except Exception as e:  # noqa
        out["error"] = str(e)
        return out


def parse_tables_arg(tables_arg: List[str], default_age: int) -> List[Dict]:
    parsed = []
    for entry in tables_arg:
        parts = entry.split(":")
        if len(parts) == 1:
            table = parts[0]
            ts_col = "updated_at"
            age = default_age
        elif len(parts) == 2:
            table, ts_col = parts
            age = default_age
        else:
            table, ts_col, age_s = parts[:3]
            try:
                age = int(age_s)
            except ValueError:
                age = default_age
        parsed.append({"table": table, "ts_column": ts_col, "max_age": age})
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Freshness monitor")
    parser.add_argument("--tables", nargs="*", default=[], help="table:timestamp_column:max_age_minutes entries")
    parser.add_argument("--default-age", type=int, default=1440, help="Default max age in minutes")
    parser.add_argument("--db-path", default=DEFAULT_DB)
    parser.add_argument("--artifact-dir", default=ARTIFACT_DIR)
    parser.add_argument("--fail-on-warn", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        print(f"Database not found: {args.db_path}", file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.artifact_dir, exist_ok=True)
    tables = parse_tables_arg(args.tables, args.default_age)
    if not tables:
        print("No tables specified; nothing to check.")
        return 0

    results = []
    overall_ok = True
    with sqlite3.connect(args.db_path) as conn:
        for spec in tables:
            res = check_table(conn, spec["table"], spec["ts_column"], spec["max_age"])
            results.append(res)
            if not res.get("ok"):
                overall_ok = False

    artifact = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "db_path": args.db_path,
        "tables": results,
        "overall_ok": overall_ok,
    }
    fname = f"freshness_{int(time.time())}.json"
    path = os.path.join(args.artifact_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    # Human summary
    print("Freshness Summary:")
    for r in results:
        status = "OK" if r.get("ok") else "STALE"
        age = r.get("age_minutes")
        age_str = f"{age}m" if age is not None else "?"
        print(f" - {r['table']}.{r['ts_column']}: age={age_str} max={r['max_age_minutes']}m => {status}")

    if not overall_ok and args.fail_on_warn:
        return 1
    return 0

if __name__ == "__main__":  # pragma: no cover
    code = main()
    sys.exit(code)
