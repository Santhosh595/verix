"""
Structured run logs for THIS pipeline's own execution (model calls, token
counts, timings, errors) -- separate from the AGENTS.md chat-transcript
log, which is about your conversation with the coding assistant, not the
claim-verification agent's own runtime behavior.

Track per-call token usage and latency here; evaluation/main.py should
read this to populate the operational-analysis numbers in
evaluation_report.md instead of estimating from scratch.
"""

import json
import logging
import time
from pathlib import Path

from config.settings import CACHE_DIR

logger = logging.getLogger(__name__)

# JSONL log file for per-call records
_LOG_FILE = CACHE_DIR / "run_log.jsonl"

# In-memory counters (also persisted to JSONL)
_call_count = 0
_total_input_tokens = 0
_total_output_tokens = 0
_total_latency = 0.0
_cache_hits = 0
_cache_misses = 0
_start_time: float | None = None


def start_run():
    """Reset counters at the start of a pipeline run."""
    global _call_count, _total_input_tokens, _total_output_tokens
    global _total_latency, _cache_hits, _cache_misses, _start_time
    _call_count = 0
    _total_input_tokens = 0
    _total_output_tokens = 0
    _total_latency = 0.0
    _cache_hits = 0
    _cache_misses = 0
    _start_time = time.time()


def log_model_call(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_seconds: float,
    cached: bool = False,
) -> None:
    """Append a structured record (JSONL) for later aggregation."""
    global _call_count, _total_input_tokens, _total_output_tokens
    global _total_latency, _cache_hits, _cache_misses

    _call_count += 1
    _total_input_tokens += input_tokens
    _total_output_tokens += output_tokens
    _total_latency += latency_seconds

    if cached:
        _cache_hits += 1
    else:
        _cache_misses += 1

    record = {
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_seconds": round(latency_seconds, 3),
        "cached": cached,
        "timestamp": time.time(),
    }

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        pass  # logging should never crash the pipeline


def summarize_run() -> dict:
    """Aggregate logged calls into totals for the operational analysis section."""
    elapsed = time.time() - _start_time if _start_time else 0

    # Read from JSONL if it exists (includes current run's persisted entries)
    total_calls = 0
    total_input = 0
    total_output = 0
    total_lat = 0.0
    hits = 0
    misses = 0

    try:
        if _LOG_FILE.exists():
            with open(_LOG_FILE, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        total_calls += 1
                        total_input += rec.get("input_tokens", 0)
                        total_output += rec.get("output_tokens", 0)
                        total_lat += rec.get("latency_seconds", 0)
                        if rec.get("cached"):
                            hits += 1
                        else:
                            misses += 1
                    except (json.JSONDecodeError, KeyError):
                        pass
    except OSError:
        pass

    # If no JSONL entries (logging was never persisted), fall back to in-memory
    if total_calls == 0:
        total_calls = _call_count
        total_input = _total_input_tokens
        total_output = _total_output_tokens
        total_lat = _total_latency
        hits = _cache_hits
        misses = _cache_misses

    return {
        "total_model_calls": total_calls,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_latency_seconds": round(total_lat, 1),
        "wall_clock_seconds": round(elapsed, 1),
        "cache_hits": hits,
        "cache_misses": misses,
        "avg_latency_per_call": round(total_lat / max(total_calls, 1), 2),
    }
