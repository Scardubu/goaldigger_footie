import logging
from datetime import datetime
from typing import Any, Dict

PUBLISHED_PREDICTIONS = []
logger = logging.getLogger(__name__)

def publish_prediction(prediction: Any, meta: Dict = None) -> None:
    """
    Central publish hook. Records prediction and metadata to in-memory list.
    Extend this to route to DB, Slack, webhook, etc.
    """
    from datetime import timezone
    entry = {
        'prediction': prediction,
        'meta': meta or {},
        'published_at': datetime.now(timezone.utc).isoformat()
    }
    PUBLISHED_PREDICTIONS.append(entry)
    logger.info(f"[PUBLISH] Prediction published: {entry}")
