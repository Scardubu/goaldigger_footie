"""Unicode-safe logging helpers for GoalDiggers platform."""
from __future__ import annotations

import logging
from typing import Any

__all__ = ["safe_log"]


_REPLACEMENTS = {
    "🎯": "[TARGET]",
    "✅": "[SUCCESS]",
    "❌": "[ERROR]",
    "⚠️": "[WARNING]",
    "🔧": "[TOOL]",
    "🚀": "[LAUNCH]",
    "💾": "[CACHE]",
    "🏆": "[TROPHY]",
    "⭐": "[STAR]",
    "🎨": "[ART]",
    "📋": "[CLIPBOARD]",
    "🔄": "[REFRESH]",
    "🌤️": "[WEATHER]",
    "🤖": "[BOT]",
    "🧮": "[CALC]",
    "🔮": "[CRYSTAL]",
    "📊": "[CHART]",
    "⚡": "[BOLT]",
    "🔥": "[FIRE]",
    "🗑️": "[TRASH]",
    "🧹": "[CLEAN]",
}


def safe_log(logger: logging.Logger, level: str, message: Any) -> None:
    """Log messages while stripping emojis/non-ascii characters."""
    try:
        clean_msg = str(message)
        for symbol, replacement in _REPLACEMENTS.items():
            clean_msg = clean_msg.replace(symbol, replacement)
        clean_msg = "".join(char for char in clean_msg if ord(char) < 128)
        getattr(logger, level)(clean_msg)
    except Exception:  # pragma: no cover - defensive fallback
        getattr(logger, level)("Logging message")
