#!/usr/bin/env python3
"""
Idempotent helper to populate `last_synced_at` for existing matches.
It attempts to set `last_synced_at` from any available `updated_at` or `fetched_at` columns.
If none exist, it leaves NULL to avoid unsafe guesses.

Usage:
    python scripts/populate_last_synced_at.py --dry-run
    python scripts/populate_last_synced_at.py
"""
import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone

DB_URL = os.getenv('DATABASE_URL')


def _connect_sqlite(path):
    return sqlite3.connect(path)


def main(dry_run: bool = True):
    # Only support local SQLite for safety in this helper. For Postgres, use DB migration tooling (alembic).
    if not DB_URL or DB_URL.startswith('sqlite'):
        # Determine sqlite path
        if DB_URL and DB_URL.startswith('sqlite:///'):
            path = DB_URL.replace('sqlite:///', '')
        else:
            # default fallback
            path = 'data/local_fallback.db'
        if not os.path.exists(path):
            print(f"Database file not found: {path}")
            return
        conn = _connect_sqlite(path)
        cur = conn.cursor()
        # Check if column exists
        try:
            cur.execute("PRAGMA table_info(matches);")
            cols = [r[1] for r in cur.fetchall()]
            if 'last_synced_at' not in cols:
                print('last_synced_at column not present. Run SQL migration first.')
                return
            # Try to infer from updated_at / fetched_at
            candidate_cols = [c for c in ('updated_at', 'fetched_at', 'synced_at') if c in cols]
            if not candidate_cols:
                print('No candidate timestamp columns found to populate last_synced_at. Leaving NULL.')
                return
            source = candidate_cols[0]
            print(f'Populating last_synced_at from {source} (dry_run={dry_run})')
            if dry_run:
                cur.execute(f"SELECT COUNT(1) FROM matches WHERE last_synced_at IS NULL AND {source} IS NOT NULL;")
                to_update = cur.fetchone()[0]
                print(f'{to_update} rows would be updated')
                return
            cur.execute(f"UPDATE matches SET last_synced_at = {source} WHERE last_synced_at IS NULL AND {source} IS NOT NULL;")
            conn.commit()
            print('Population complete')
        finally:
            conn.close()
    else:
        print('This helper only supports SQLite local operations by default. For Postgres use your migration tool (alembic/psql).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Show changes without applying')
    args = parser.parse_args()
    main(dry_run=args.dry_run)
