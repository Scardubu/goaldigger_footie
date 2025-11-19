import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def _to_epoch(ts) -> Optional[float]:
    """Convert common timestamp formats to epoch seconds (UTC).

    Supports numeric epoch, datetime, and ISO-like strings. Returns None on failure.
    """
    if ts is None:
        return None
    try:
        # Numeric timestamp
        if isinstance(ts, (int, float)):
            return float(ts)

        # datetime object
        if hasattr(ts, 'timestamp') and isinstance(ts, datetime):
            dt = ts
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return float(dt.timestamp())

        # ISO string parsing (best-effort)
        if isinstance(ts, str):
            s = ts.strip()
            # Handle trailing Z
            if s.endswith('Z'):
                s = s[:-1]
            try:
                dt = datetime.fromisoformat(s)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return float(dt.timestamp())
            except Exception:
                # Try a couple common formats
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
                    try:
                        dt = datetime.strptime(s, fmt)
                        dt = dt.replace(tzinfo=timezone.utc)
                        return float(dt.timestamp())
                    except Exception:
                        continue
    except Exception as e:
        logger.debug(f"_to_epoch parse error: {e}")

    return None


def is_fresh(ts, max_age_hours: float = 24.0) -> bool:
    """Return True if timestamp `ts` is within `max_age_hours` from now.

    If `ts` cannot be parsed, returns False.
    """
    epoch = _to_epoch(ts)
    if epoch is None:
        return False
    now = datetime.now(timezone.utc).timestamp()
    age_seconds = now - float(epoch)
    return age_seconds <= max_age_hours * 3600.0


def freshness_summary(ts) -> str:
    """Return a human-friendly freshness summary like '2 hours ago' or 'unknown'."""
    epoch = _to_epoch(ts)
    if epoch is None:
        return "unknown"
    now = datetime.now(timezone.utc).timestamp()
    delta = int(now - epoch)
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{delta // 60} minutes ago"
    if delta < 86400:
        return f"{delta // 3600} hours ago"
    return f"{delta // 86400} days ago"


def format_timestamp(ts) -> str:
    """Return an ISO-like UTC string for known timestamps, else str(ts)."""
    epoch = _to_epoch(ts)
    if epoch is None:
        return str(ts)
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
