"""
Cache VLM/LLM responses by a content-derived key so re-running the
pipeline (during dev, or after a crash) doesn't re-spend tokens on
unchanged inputs. Keep it simple -- a local JSON file under
config.settings.CACHE_DIR is enough for a 24h hackathon; no need for a
distributed cache.
"""

import hashlib
import json
import logging
from pathlib import Path

from config.settings import CACHE_DIR

logger = logging.getLogger(__name__)


def _cache_path(key: str) -> Path:
    """Map a cache key to a file path under CACHE_DIR."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"{safe_name}.json"


def make_cache_key(*parts: str) -> str:
    """
    Stable hash of (image_hash, prompt_version, model_name, ...).
    Pass any number of string parts that uniquely identify the call.
    """
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()


def get_cached(key: str) -> dict | None:
    """Return cached response dict if present and valid, else None."""
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        # Basic shape check
        if "response" not in data and "error" not in data:
            return None
        logger.debug("Cache hit: %s", key)
        return data
    except (json.JSONDecodeError, OSError):
        # Corrupt cache entry -- treat as miss
        logger.warning("Corrupt cache entry %s, treating as miss", path)
        return None


def set_cached(key: str, value: dict) -> None:
    """Persist response dict under CACHE_DIR."""
    path = _cache_path(key)
    # Write atomically (write to temp, then rename) to survive crashes mid-write
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(value, fh, ensure_ascii=False)
        tmp.replace(path)
        logger.debug("Cache write: %s", key)
    except OSError as exc:
        logger.warning("Failed to cache response: %s", exc)


def clear_cache() -> int:
    """Remove all cached entries. Returns count of files removed."""
    if not CACHE_DIR.exists():
        return 0
    count = 0
    for p in CACHE_DIR.glob("*.json"):
        try:
            p.unlink()
            count += 1
        except OSError:
            pass
    return count
