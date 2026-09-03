#!/usr/bin/env python3
"""
Provider fallback tests for verix vision analyzer.

This module tests the automatic provider fallback chain in vision_analyzer.py:
- gemini → groq
- groq → openrouter
- openrouter → groq
- successful direct calls (no fallback)
- all providers failing
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add the code directory to the path so we can import verix modules
sys.path.insert(0, str(Path(__file__).parent / "code"))

from core.vision_analyzer import analyze_image
# Use localhost path for patch since that's where verix code runs
VISUAL_ANALYZER_PATH = "core.vision_analyzer"
@pytest.fixture
def mock_cache(monkeypatch):
    """Mock caching for deterministic test behavior."""
    def mock_make_cache_key(img_hash, prompt_version, provider, prompt_text):
        return f"{img_hash}_{prompt_version}_{provider}_{hash(prompt_text)}"
    def mock_get_cached(key):
        return None
    def mock_set_cached(key, value):
        pass

    monkeypatch.setattr(sys.modules[VISUAL_ANALYZER_PATH + ".utils.caching"], 
                       "make_cache_key", mock_make_cache_key)
    monkeypatch.setattr(sys.modules[VISUAL_ANALYZER_PATH + ".utils.caching"], 
                       "get_cached", mock_get_cached)
    monkeypatch.setattr(sys.modules[VISUAL_ANALYZER_PATH + ".utils.caching"], 
                       "set_cached", mock_set_cached)
@pytest.fixture
def mock_image_path(tmp_path):
    """Create a mock image path for testing."""
    return tmp_path / "test.jpg"
@pytest.fixture
def mock_config(monkeypatch):
    """Mock configuration for deterministic testing."""
    monkeypatch.setattr(sys.modules[VISUAL_ANALYZER_PATH + ".config.settings"],
                       "VLM_PROVIDER", "gemini")
    monkeypatch.setattr(sys.modules[VISUAL_ANALYZER_PATH + ".config.settings"],
                       "VLM_MODEL", "gemini-2.0-flash")
    monkeypatch.setattr(sys.modules[VISUAL_ANALYZER_PATH + ".config.settings"],
                       "GEMINI_API_KEY", "test_gemini_key")
    monkeypatch.setattr(sys.modules[VISUAL_ANALYZER_PATH + ".config.settings"],
                       "GROQ_API_KEY", "test_groq_key")
    monkeypatch.setattr(sys.modules[VISUAL_ANALYZER_PATH + ".config.settings"],
                       "OPENROUTER_API_KEY", "test_openrouter_key")
    monkeypatch.setattr(sys.modules[VISUAL_ANALYZER_PATH + ".config.settings"],
                       "TEMPERATURE", 0.1)
    monkeypatch.setattr(sys.modules[VISUAL_ANALYZER_PATH + ".config.settings"],
                       "MAX_RETRIES", 1)
    monkeypatch.setattr(sys.modules[VISUAL_ANALYZER_PATH + ".config.settings"],
                       "REQUEST_TIMEOUT_SECONDS", 30)
def test_gemini_to_groq_fallback(mock_cache, mock_image_path, mock_config):
    """
    Test gemini → groq fallback chain.
    
    When gemini fails, should automatically fall back to groq.
    """
    # Create mock image file
    mock_image_path.write_bytes(b"fake image content")
    
    # Mock the _call_gemini function to raise an exception (simulating failure)
    with patch(f"{VISUAL_ANALYZER_PATH}._call_gemini") as mock_gemini_call:
        mock_gemini_call.side_effect = Exception("Gemini API failed")
        
        # Mock the _call_groq function to return a successful response
        with patch(f"{VISUAL_ANALYZER_PATH}._call_groq") as mock_groq_call:
            # Mock successful groq response
            mock_groq_response = '{"issue_type": "broken", "object_part": "panel", "severity": "high", "quality_notes": [], "authenticity_notes": [], "confidence": 0.8, "description": "I see a broken car panel"}'
            mock_groq_call.return_value = mock_groq_response
            
            # Call analyze_image
            result = analyze_image(
                image_path=mock_image_path,
                claim_object="car",
                claimed_issue_summary="The car has a broken panel"
            )
            
            # Verify gemini was called first (provider is gemini)
            mock_gemini_call.assert_called_once()
            
            # Verify groq was called as fallback after gemini failure
            mock_groq_call.assert_called_once()
            
            # Verify the result came from groq (since gemini failed)
            assert result.issue_type == "broken"
            assert result.object_part == "panel"
            assert result.severity == "high"
            assert result.confidence == 0.8
            assert result.quality_notes == []
            assert result.authenticity_notes == []
def test_groq_to_openrouter_fallback(mock_cache, mock_image_path, mock_config):
    """
    Test groq → openrouter fallback chain.
    
    When groq fails, should automatically fall back to openrouter.
    """
    # Update config to start with groq provider
    import sys
    sys.modules[VISUAL_ANALYZER_PATH + ".config.settings"].VLM_PROVIDER = "groq"
    
    # Create mock image file
    mock_image_path.write_bytes(b"fake image content")
    
    # Mock the _call_groq function to raise an exception (simulating failure)
    with patch(f"{VISUAL_ANALYZER_PATH}._call_groq") as mock_groq_call:
        mock_groq_call.side_effect = Exception("Groq API failed")
        
        # Mock the _call_openrouter function to return a successful response
        with patch(f"{VISUAL_ANALYZER_PATH}._call_openrouter") as mock_openrouter_call:
            # Mock successful openrouter response
            mock_openrouter_response = '{"issue_type": "scratch", "object_part": "door", "severity": "medium", "quality_notes": ["good lighting"], "authenticity_notes": [], "confidence": 0.7, "description": "I see a scratch on the door"}'
            mock_openrouter_call.return_value = mock_openrouter_response
            
            # Mock _call_gemini for secondary fallback
            with patch(f"{VISUAL_ANALYZER_PATH}._call_gemini") as mock_gemini_call:
                # Call analyze_image
                result = analyze_image(
                    image_path=mock_image_path,
                    claim_object="car",
                    claimed_issue_summary="The car has a scratch on the door"
                )
                
                # Verify groq was called first
                mock_groq_call.assert_called_once()
                
                # Verify openrouter was called as fallback after groq failure
                mock_openrouter_call.assert_called_once()
                
                # Verify result came from openrouter
                assert result.issue_type == "scratch"
                assert result.object_part == "door"
                assert result.severity == "medium"
                assert result.quality_notes == ["good lighting"]
                assert result.authenticity_notes == []
                assert result.confidence == 0.7
def test_openrouter_to_groq_fallback(mock_cache, mock_image_path, mock_config):
    """
    Test openrouter → groq fallback chain.
    
    When openrouter fails, should automatically fall back to groq.
    """
    # Update config to start with openrouter provider
    import sys
    sys.modules[VISUAL_ANALYZER_PATH + ".config.settings"].VLM_PROVIDER = "openrouter"
    
    # Create mock image file
    mock_image_path.write_bytes(b"fake image content")
    
    # Mock the _call_openrouter function to raise an exception (simulating failure)
    with patch(f"{VISUAL_ANALYZER_PATH}._call_openrouter") as mock_openrouter_call:
        mock_openrouter_call.side_effect = Exception("OpenRouter API failed")
        
        # Mock the _call_groq function to return a successful response
        with patch(f"{VISUAL_ANALYZER_PATH}._call_groq") as mock_groq_call:
            # Mock successful groq response
            mock_groq_response = '{"issue_type": "crack", "object_part": "windshield", "severity": "high", "quality_notes": [], "authenticity_notes": ["clear"], "confidence": 0.9, "description": "I see a crack in the windshield"}'
            mock_groq_call.return_value = mock_groq_response
            
            # Mock _call_gemini for completeness
            with patch(f"{VISUAL_ANALYZER_PATH}._call_gemini") as mock_gemini_call:
                # Call analyze_image
                result = analyze_image(
                    image_path=mock_image_path,
                    claim_object="car",
                    claimed_issue_summary="The car has a crack in the windshield"
                )
                
                # Verify openrouter was called first
                mock_openrouter_call.assert_called_once()
                
                # Verify groq was called as fallback after openrouter failure
                mock_groq_call.assert_called_once()
                
                # Verify result came from groq
                assert result.issue_type == "crack"
                assert result.object_part == "windshield"
                assert result.severity == "high"
                assert result.quality_notes == []
                assert result.authenticity_notes == ["clear"]
                assert result.confidence == 0.9
def test_successful_direct_call_no_fallback(mock_cache, mock_image_path, mock_config):
    """
    Test that when the primary provider succeeds, no fallback occurs.
    """
    # Create mock image file
    mock_image_path.write_bytes(b"fake image content")
    
    # Mock the _call_gemini function to succeed
    with patch(f"{VISUAL_ANALYZER_PATH}._call_gemini") as mock_gemini_call:
        # Mock successful gemini response
        mock_gemini_response = '{"issue_type": "dent", "object_part": "door", "severity": "medium", "quality_notes": ["needs closer look"], "authenticity_notes": [], "confidence": 0.6, "description": "I see a dent in the door"}'
        mock_gemini_call.return_value = mock_gemini_response
        
        # Mock _call_groq to ensure it's not called
        with patch(f"{VISUAL_ANALYZER_PATH}._call_groq") as mock_groq_call:
            # Call analyze_image
            result = analyze_image(
                image_path=mock_image_path,
                claim_object="car",
                claimed_issue_summary="The car has a dent in the door"
            )
            
            # Verify gemini was called
            mock_gemini_call.assert_called_once()
            
            # Verify groq was NOT called (no fallback needed)
            mock_groq_call.assert_not_called()
            
            # Verify result came from gemini
            assert result.issue_type == "dent"
            assert result.object_part == "door"
            assert result.severity == "medium"
            assert result.quality_notes == ["needs closer look"]
            assert result.authenticity_notes == []
            assert result.confidence == 0.6
def test_fallback_chain_all_fail(mock_cache, mock_image_path, mock_config):
    """
    Test when all providers in the fallback chain fail, expect error.
    """
    # Create mock image file
    mock_image_path.write_bytes(b"fake image content")
    
    # Mock all provider calls to fail
    with patch(f"{VISUAL_ANALYZER_PATH}._call_gemini") as mock_gemini_call:
        mock_gemini_call.side_effect = Exception("Gemini API failed")
        
        with patch(f"{VISUAL_ANALYZER_PATH}._call_groq") as mock_groq_call:
            mock_groq_call.side_effect = Exception("Groq API failed")
            
            with patch(f"{VISUAL_ANALYZER_PATH}._call_openrouter") as mock_openrouter_call:
                mock_openrouter_call.side_effect = Exception("OpenRouter API failed")
                
                # Expect an exception when all providers fail
                with pytest.raises(Exception, match="VLM call failed after 1 retries"):
                    analyze_image(
                        image_path=mock_image_path,
                        claim_object="car",
                        claimed_issue_summary="Test claim"
                    )