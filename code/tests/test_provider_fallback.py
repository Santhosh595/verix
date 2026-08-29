"""
Tests for the VLM provider-fallback dispatch in core/vision_analyzer.py.

These tests assert the documented fallback chain without any network access
and without any real API keys:

    gemini   -> groq
    groq     -> openrouter   (via VLM_FALLBACK_MODEL)
    openrouter -> groq

and the default provider/model resolution declared in config/settings.py
(openrouter / openai/gpt-oss-120b:free).

The real provider call functions (_call_gemini/_call_groq/_call_openrouter)
are replaced with recording fakes via monkeypatch, so no SDK import or
network call ever happens. The module-level key globals are also patched
so the key-presence gates in _call_vlm behave as if keys were configured.
"""

import pytest

from config import settings as _settings
from core import vision_analyzer as va


class FakeCaller:
    """Callable stand-in for a provider call function.

    Records each invocation as (image_b64, prompt, model_name) and either
    raises a configured exception (to simulate a failing provider) or
    returns a fixed string.
    """

    def __init__(self, side_effect=None, return_value="OK"):
        self.calls = []
        self.side_effect = side_effect
        self.return_value = return_value

    def __call__(self, image_b64, prompt, model_name=None):
        self.calls.append((image_b64, prompt, model_name))
        if self.side_effect is not None:
            raise self.side_effect
        return self.return_value

    @property
    def call_count(self):
        return len(self.calls)


@pytest.fixture(autouse=True)
def isolated_dispatch(monkeypatch):
    """Patch provider callers + key globals + retry count for each test."""
    monkeypatch.setattr(va, "MAX_RETRIES", 1)
    monkeypatch.setattr(va, "GEMINI_API_KEY", None)
    monkeypatch.setattr(va, "GROQ_API_KEY", None)
    monkeypatch.setattr(va, "OPENAI_API_KEY", None)
    monkeypatch.setattr(va, "ANTHROPIC_API_KEY", None)
    monkeypatch.setattr(va, "OPENROUTER_API_KEY", None)
    # Install recording fakes for the three fallback-relevant providers.
    gemini = FakeCaller()
    groq = FakeCaller()
    openrouter = FakeCaller()
    monkeypatch.setattr(va, "_call_gemini", gemini)
    monkeypatch.setattr(va, "_call_groq", groq)
    monkeypatch.setattr(va, "_call_openrouter", openrouter)
    ctx = {"gemini": gemini, "groq": groq, "openrouter": openrouter}
    yield ctx


# ---------------------------------------------------------------------------
# Default provider / model resolution (settings.py:41-43)
# ---------------------------------------------------------------------------

def test_default_provider_resolution():
    """The configured default must be openrouter / openai-gpt-oss-120b:free."""
    assert _settings.VLM_PROVIDER == "openrouter"
    assert _settings.VLM_MODEL == "openai/gpt-oss-120b:free"
    assert _settings.VLM_FALLBACK_MODEL == "deepseek/deepseek-chat:free"


def test_default_dispatch_uses_openrouter_without_fallback(isolated_dispatch):
    """With no provider override and only the OpenRouter key set, the call
    is dispatched straight to OpenRouter and never falls back to Groq."""
    ctx = isolated_dispatch
    va.OPENROUTER_API_KEY = "ort-key"
    ctx["openrouter"].return_value = "ORT-RESULT"

    result = va._call_vlm("b64img", "a prompt")

    assert result == "ORT-RESULT"
    assert ctx["openrouter"].call_count == 1
    assert ctx["groq"].call_count == 0
    assert ctx["gemini"].call_count == 0


# ---------------------------------------------------------------------------
# Fallback hops (vision_analyzer.py:366-375)
# ---------------------------------------------------------------------------

def test_gemini_falls_back_to_groq(isolated_dispatch):
    """gemini primary fails -> Groq is used on fallback."""
    ctx = isolated_dispatch
    va.GEMINI_API_KEY = "gem-key"
    va.GROQ_API_KEY = "groq-key"
    ctx["gemini"].side_effect = RuntimeError("gemini down")
    ctx["groq"].return_value = "GROQ-RESULT"

    result = va._call_vlm("b64img", "a prompt", provider="gemini")

    assert result == "GROQ-RESULT"
    # Primary attempted MAX_RETRIES times before fallback.
    assert ctx["gemini"].call_count == va.MAX_RETRIES
    assert ctx["groq"].call_count == 1
    # Groq fallback ignores VLM_MODEL and uses its own default.
    assert ctx["groq"].calls[0][2] is None


def test_groq_falls_back_to_openrouter(isolated_dispatch):
    """groq primary fails -> OpenRouter is used, with VLM_FALLBACK_MODEL."""
    ctx = isolated_dispatch
    va.GROQ_API_KEY = "groq-key"
    va.OPENROUTER_API_KEY = "ort-key"
    ctx["groq"].side_effect = RuntimeError("groq down")
    ctx["openrouter"].return_value = "ORT-RESULT"

    result = va._call_vlm("b64img", "a prompt", provider="groq")

    assert result == "ORT-RESULT"
    assert ctx["groq"].call_count == va.MAX_RETRIES
    assert ctx["openrouter"].call_count == 1
    # OpenRouter fallback explicitly passes VLM_FALLBACK_MODEL.
    assert ctx["openrouter"].calls[0][2] == _settings.VLM_FALLBACK_MODEL


def test_openrouter_falls_back_to_groq(isolated_dispatch):
    """openrouter primary fails -> Groq is used on fallback."""
    ctx = isolated_dispatch
    va.OPENROUTER_API_KEY = "ort-key"
    va.GROQ_API_KEY = "groq-key"
    ctx["openrouter"].side_effect = RuntimeError("openrouter down")
    ctx["groq"].return_value = "GROQ-RESULT"

    result = va._call_vlm("b64img", "a prompt", provider="openrouter")

    assert result == "GROQ-RESULT"
    assert ctx["openrouter"].call_count == va.MAX_RETRIES
    assert ctx["groq"].call_count == 1


# ---------------------------------------------------------------------------
# No-fallback / failure cases
# ---------------------------------------------------------------------------

def test_gemini_no_fallback_when_groq_key_missing(isolated_dispatch):
    """gemini fails but Groq key absent -> no fallback, RuntimeError raised."""
    ctx = isolated_dispatch
    va.GEMINI_API_KEY = "gem-key"
    # GROQ_API_KEY deliberately left None.
    ctx["gemini"].side_effect = RuntimeError("gemini down")

    with pytest.raises(RuntimeError, match="VLM call failed after"):
        va._call_vlm("b64img", "a prompt", provider="gemini")

    assert ctx["groq"].call_count == 0


def test_openrouter_no_fallback_when_groq_key_missing(isolated_dispatch):
    """openrouter fails but Groq key absent -> no fallback, error raised."""
    ctx = isolated_dispatch
    va.OPENROUTER_API_KEY = "ort-key"
    # GROQ_API_KEY deliberately left None.
    ctx["openrouter"].side_effect = RuntimeError("openrouter down")

    with pytest.raises(RuntimeError, match="VLM call failed after"):
        va._call_vlm("b64img", "a prompt", provider="openrouter")

    assert ctx["groq"].call_count == 0


def test_missing_api_key_raises_before_retries(isolated_dispatch):
    """No key for the chosen provider -> immediate RuntimeError, no calls."""
    ctx = isolated_dispatch
    # No keys set at all.

    with pytest.raises(RuntimeError, match="No API key for provider 'groq'"):
        va._call_vlm("b64img", "a prompt", provider="groq")

    assert ctx["groq"].call_count == 0


def test_unknown_provider_raises(isolated_dispatch):
    """An unsupported provider name is rejected by the dispatcher."""
    with pytest.raises(ValueError, match="Unknown VLM_PROVIDER"):
        va._call_vlm("b64img", "a prompt", provider="not-a-real-provider")
