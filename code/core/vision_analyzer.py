"""
VLM calls. This is the only module that should "look at pixels". Keep it
provider-agnostic at the call site (config.settings.VLM_PROVIDER) so
evaluation/compare_strategies.py can swap providers/prompts cleanly.

Primary: Google Gemini (google-genai SDK)
Fallback: Groq (groq SDK, OpenAI-compatible)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from config.schema import (
    ISSUE_TYPES,
    OBJECT_PARTS,
    SEVERITIES,
    closest_allowed_value,
)
from config.settings import (
    VLM_PROVIDER,
    VLM_MODEL,
    VLM_FALLBACK_MODEL,
    GEMINI_API_KEY,
    GROQ_API_KEY,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    OPENROUTER_API_KEY,
    TEMPERATURE,
    MAX_RETRIES,
    REQUEST_TIMEOUT_SECONDS,
)
from utils.caching import make_cache_key, get_cached, set_cached
from utils.image_utils import load_and_encode, cheap_quality_heuristics, image_hash

logger = logging.getLogger(__name__)

PROMPT_VERSION = "1.0"  # bump when prompt template changes, invalidates cache


@dataclass
class ImageFinding:
    image_id: str
    issue_type: str          # one of config.schema.ISSUE_TYPES
    object_part: str         # one of config.schema.OBJECT_PARTS[claim_object]
    severity: str            # one of config.schema.SEVERITIES
    quality_notes: list[str] = field(default_factory=list)
    authenticity_notes: list[str] = field(default_factory=list)
    confidence: float = 0.5
    raw_model_output: str = ""
    description: str = ""    # free-text description from model (not used in output, kept for debugging)


def _build_vision_prompt(claim_object: str) -> str:
    """Build the user prompt for vision analysis, injecting closed vocabularies."""
    return (
        f"Object type: {claim_object}\n"
        "What the user claims happened (for context only -- verify, don't assume):\n"
        '"""\n'
        "{claimed_issue_summary}\n"
        '"""\n'
        f"\nAllowed issue_type values: {ISSUE_TYPES}\n"
        f"Allowed object_part values for {claim_object}: {OBJECT_PARTS[claim_object]}\n"
        f"Allowed severity values: {SEVERITIES}\n\n"
        "Respond as JSON with keys: issue_type, object_part, severity, "
        "quality_notes (list of short strings), authenticity_notes (list of short "
        "strings), confidence (0-1), description (1-2 sentences of what you actually see)."
    )


def _parse_model_json(raw: str) -> dict:
    """Extract JSON object from model output (handles markdown fences, etc.)."""
    # Strip markdown code fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)

    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try to find first {...} block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from model output: {raw[:200]}")


def _validate_finding(finding: dict, claim_object: str) -> dict:
    """Normalize and validate parsed JSON against closed vocabularies."""
    issue_type = closest_allowed_value(
        finding.get("issue_type", "unknown"), ISSUE_TYPES, default="unknown"
    )
    object_part = closest_allowed_value(
        finding.get("object_part", "unknown"),
        OBJECT_PARTS.get(claim_object, ["unknown"]),
        default="unknown",
    )
    severity = closest_allowed_value(
        finding.get("severity", "unknown"), SEVERITIES, default="unknown"
    )

    quality_notes = finding.get("quality_notes", [])
    if isinstance(quality_notes, str):
        quality_notes = [quality_notes]
    elif not isinstance(quality_notes, list):
        quality_notes = []
    authenticity_notes = finding.get("authenticity_notes", [])
    if isinstance(authenticity_notes, str):
        authenticity_notes = [authenticity_notes]
    elif not isinstance(authenticity_notes, list):
        authenticity_notes = []

    confidence = finding.get("confidence", 0.5)
    try:
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))
    except (ValueError, TypeError):
        confidence = 0.5

    return {
        "issue_type": issue_type,
        "object_part": object_part,
        "severity": severity,
        "quality_notes": quality_notes,
        "authenticity_notes": authenticity_notes,
        "confidence": confidence,
        "description": finding.get("description", ""),
    }


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------

def _call_gemini(
    image_b64: str,
    prompt: str,
    model_name: str | None = None,
) -> str:
    """Call Google Gemini vision API. Returns raw text response."""
    import google.genai as genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    model = model_name or VLM_MODEL or "gemini-2.0-flash"

    response = client.models.generate_content(
        model=model,
        contents=[
            prompt,
            types.Part(inline_data=types.Blob(data=image_b64, mime_type="image/jpeg")),
        ],
        config={"temperature": TEMPERATURE},
    )
    return response.text or ""


def _call_groq(
    image_b64: str,
    prompt: str,
    model_name: str | None = None,
) -> str:
    """Call Groq vision API (OpenAI-compatible). Returns raw text response."""
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    # Groq fallback always uses a Groq-hosted model, ignoring VLM_MODEL
    model = model_name or "meta-llama/llama-4-scout-17b-16e-instruct"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
        temperature=TEMPERATURE,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


def _call_openai(
    image_b64: str,
    prompt: str,
    model_name: str | None = None,
) -> str:
    """Call OpenAI vision API. Returns raw text response."""
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    model = model_name or VLM_MODEL or "gpt-4o-mini"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
        temperature=TEMPERATURE,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


def _call_openrouter(
    image_b64: str,
    prompt: str,
    model_name: str | None = None,
) -> str:
    """Call OpenRouter API (OpenAI-compatible). Returns raw text response.

    Supports any OpenRouter model including free ones with :free suffix.
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    model = model_name or VLM_MODEL or "nex-agi/nex-n2-pro:free"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
        temperature=TEMPERATURE,
        max_tokens=512,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    return response.choices[0].message.content or ""


def _call_anthropic(
    image_b64: str,
    prompt: str,
    model_name: str | None = None,
) -> str:
    """Call Anthropic Claude vision API. Returns raw text response."""
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    model = model_name or VLM_MODEL or "claude-3-5-sonnet-20241022"

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=TEMPERATURE,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return "\n".join(block.text for block in response.content if hasattr(block, "text"))


def _get_caller(provider: str):
    """Return the call function for a given provider name."""
    dispatch = {
        "gemini": _call_gemini,
        "groq": _call_groq,
        "openai": _call_openai,
        "openrouter": _call_openrouter,
        "anthropic": _call_anthropic,
    }
    caller = dispatch.get(provider.lower())
    if caller is None:
        raise ValueError(
            f"Unknown VLM_PROVIDER '{provider}'. "
            f"Supported: {list(dispatch.keys())}"
        )
    return caller


def _get_api_key(provider: str) -> str | None:
    """Return the API key env var for a given provider."""
    keys = {
        "gemini": GEMINI_API_KEY,
        "groq": GROQ_API_KEY,
        "openai": OPENAI_API_KEY,
        "openrouter": OPENROUTER_API_KEY,
        "anthropic": ANTHROPIC_API_KEY,
    }
    return keys.get(provider.lower())


def _call_vlm(
    image_b64: str,
    prompt: str,
    provider: str | None = None,
    model_name: str | None = None,
) -> str:
    """
    Dispatch a VLM call. Tries the configured provider first; if it fails
    and the provider is 'gemini', falls back to Groq automatically.
    Retries on transient errors up to MAX_RETRIES.
    """
    provider = (provider or VLM_PROVIDER).lower()
    caller = _get_caller(provider)
    api_key = _get_api_key(provider)

    if not api_key:
        raise RuntimeError(
            f"No API key for provider '{provider}'. "
            f"Set the corresponding env variable."
        )

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return caller(image_b64, prompt, model_name)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "VLM call attempt %d/%d failed (%s): %s",
                attempt, MAX_RETRIES, provider, exc,
            )

    # Provider fallback chain: if primary fails, try alternatives
    if provider == "gemini" and GROQ_API_KEY:
        logger.info("Gemini failed after %d retries, falling back to Groq", MAX_RETRIES)
        return _call_groq(image_b64, prompt, model_name)
    if provider == "groq" and OPENROUTER_API_KEY:
        logger.info("Groq failed after %d retries, falling back to OpenRouter", MAX_RETRIES)
        return _call_openrouter(image_b64, prompt, VLM_FALLBACK_MODEL)
    if provider == "openrouter" and GROQ_API_KEY:
        logger.info("OpenRouter failed after %d retries, falling back to Groq", MAX_RETRIES)
        return _call_groq(image_b64, prompt, model_name)

    raise RuntimeError(
        f"VLM call failed after {MAX_RETRIES} retries ({provider}): {last_exc}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_image(
    image_path: Path,
    claim_object: str,
    claimed_issue_summary: str,
) -> ImageFinding:
    """
    Analyze a single image against a claim. Encodes the image, sends it to
    the configured VLM with a structured prompt, and validates the response
    against closed vocabularies. Caches by (image hash, prompt version,
    provider).
    """
    image_id = image_path.stem

    # Cheap heuristics first (these go into quality_notes, not cache key)
    heuristics = cheap_quality_heuristics(image_path) if image_path.exists() else ["missing"]

    # Build cache key
    img_hash = image_hash(image_path) if image_path.exists() else "missing"
    prompt_text = _build_vision_prompt(claim_object)
    cache_key = make_cache_key(img_hash, PROMPT_VERSION, VLM_PROVIDER, prompt_text)

    # Check cache
    cached = get_cached(cache_key)
    if cached is not None:
        finding_data = cached.get("response", {})
        finding_data = _validate_finding(finding_data, claim_object)
        finding_data["quality_notes"] = list(
            set(finding_data.get("quality_notes", []) + heuristics)
        )
        return ImageFinding(
            image_id=image_id,
            raw_model_output=finding_data.get("raw", ""),
            **finding_data,
        )

    # Encode image
    try:
        image_b64 = load_and_encode(image_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Cannot encode image %s: %s", image_path, exc)
        return ImageFinding(
            image_id=image_id,
            issue_type="unknown",
            object_part="unknown",
            severity="unknown",
            quality_notes=heuristics + ["unreadable"],
            authenticity_notes=[],
            confidence=0.0,
            raw_model_output="",
        )

    # Call VLM
    raw_response = _call_vlm(image_b64, prompt_text)

    # Parse and validate
    try:
        finding_data = _parse_model_json(raw_response)
        finding_data = _validate_finding(finding_data, claim_object)
    except (ValueError, KeyError) as exc:
        logger.warning("Could not parse VLM response for %s: %s", image_path, exc)
        return ImageFinding(
            image_id=image_id,
            issue_type="unknown",
            object_part="unknown",
            severity="unknown",
            quality_notes=heuristics,
            authenticity_notes=[],
            confidence=0.2,
            raw_model_output=raw_response[:500],
        )

    # Merge heuristics into quality_notes
    finding_data["quality_notes"] = list(
        set(finding_data.get("quality_notes", []) + heuristics)
    )

    # Cache the result
    set_cached(cache_key, {"response": finding_data, "raw": raw_response[:500]})

    return ImageFinding(
        image_id=image_id,
        raw_model_output=raw_response[:500],
        **finding_data,
    )


def analyze_claim_images(
    image_paths: list[Path],
    claim_object: str,
    claimed_issue_summary: str,
) -> list[ImageFinding]:
    """
    Analyze each image individually. Per-image calls give us granular
    supporting_image_ids (which images actually showed the claimed issue).
    A multi-image strategy can be compared in evaluation/compare_strategies.py.
    """
    findings: list[ImageFinding] = []
    for path in image_paths:
        finding = analyze_image(path, claim_object, claimed_issue_summary)
        findings.append(finding)
    return findings
