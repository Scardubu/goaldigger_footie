import sqlite3
import time
from datetime import datetime, timezone


def check_table_freshness(db_path, table, ts_col, max_age_minutes):
    try:
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.execute(f"SELECT {ts_col} FROM {table} WHERE {ts_col} IS NOT NULL ORDER BY {ts_col} DESC LIMIT 1")
            row = cur.fetchone()
            if not row:
                return {"is_fresh": False, "table": table, "age_minutes": None, "status": "no data"}
            
            latest = row[0]
            
            # Try parsing different timestamp formats
            ts_val = None
            try:
                # ISO format
                if isinstance(latest, str) and 'T' in latest:
                    dt = datetime.fromisoformat(latest.replace('Z', '+00:00'))
                    ts_val = dt.timestamp()
                # Standard datetime string
                elif isinstance(latest, str):
                    ts_val = time.mktime(time.strptime(str(latest)[:19], "%Y-%m-%d %H:%M:%S"))
                # Unix timestamp
                elif isinstance(latest, (int, float)):
                    ts_val = float(latest)
                    if ts_val > 1e12:  # Milliseconds
                        ts_val /= 1000.0
            except Exception:
                return {"is_fresh": False, "table": table, "age_minutes": None, "status": "unparseable timestamp"}
            
            if ts_val is None:
                return {"is_fresh": False, "table": table, "age_minutes": None, "status": "unparseable timestamp"}
            
            age_minutes = (time.time() - ts_val) / 60.0
            return {
                "is_fresh": age_minutes <= max_age_minutes, 
                "table": table, 
                "age_minutes": round(age_minutes, 2), 
                "status": "ok"
            }
        finally:
            conn.close()
    except sqlite3.OperationalError as e:
        return {"is_fresh": False, "table": table, "age_minutes": None, "status": f"database error: {str(e)}"}
    except Exception as e:
        return {"is_fresh": False, "table": table, "age_minutes": None, "status": f"error: {str(e)}"}

class FreshnessMonitor:
    def __init__(self, db_path, output_dir):
        self.db_path = db_path
        self.output_dir = output_dir
    
    def run(self, checks):
        import json
        import os
        import time
        from pathlib import Path

        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        results = []
        for check in checks:
            result = check_table_freshness(self.db_path, check["table"], check["timestamp_column"], check["max_age_minutes"])
            results.append(result)
        
        all_fresh = all(r.get("is_fresh", False) for r in results)
        worst_age = max((r.get("age_minutes", 0) for r in results if r.get("age_minutes") is not None), default=0)
        
        # Save artifact
        artifact = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "db_path": self.db_path,
            "results": results,
            "overall_status": "fresh" if all_fresh else "stale",
            "worst_age_minutes": worst_age
        }
        
        timestamp = int(time.time())
        artifact_path = Path(self.output_dir) / f"freshness_check_{timestamp}.json"
        with open(artifact_path, 'w') as f:
            json.dump(artifact, f, indent=2)
        
        return 0 if all_fresh else 1

__all__ = ["FreshnessMonitor", "check_table_freshness"]
