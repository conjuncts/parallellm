"""
Unit tests for LLMIdentity and provider guessing functionality

Tests the LLM identity system including:
- LLMIdentity creation and provider detection
- Provider guessing from identity strings
- String conversion for different providers
- Default provider behavior
"""

import pytest
from parallellm.core.response import LLMIdentity
from parallellm.provider.guess import guess_provider


class TestLLMIdentity:
    """Test LLMIdentity creation and provider handling"""

    def test_identity_creation_with_guessed_provider(self):
        """Test creating identity with provider guessing"""
        identity = LLMIdentity("gpt-4")

        assert identity.identity == "gpt-4"
        assert identity.provider == "openai"  # Should be guessed from gpt- prefix

    def test_identity_creation_unknown_provider(self):
        """Test creating identity with unknown provider"""
        identity = LLMIdentity("unknown-model")

        assert identity.identity == "unknown-model"
        assert identity.provider is None  # Cannot be guessed

    def test_to_str_with_explicit_provider(self):
        """Test string conversion with explicit provider"""
        identity = LLMIdentity("gpt-4", provider="openai")

        result = identity.to_str(provider="openai")
        assert result == "gpt-4"

    def test_to_str_with_inferred_provider(self):
        """Test string conversion using inferred provider"""
        identity = LLMIdentity("gpt-3.5-turbo")  # Should auto-detect openai

        result = identity.to_str()  # No explicit provider
        assert result == "gpt-3.5-turbo"

    def test_to_str_with_none_identity_openai_default(self):
        """Test string conversion with None identity defaults to openai model"""
        identity = LLMIdentity("gpt-4")
        identity.identity = None  # Simulate None identity

        result = identity.to_str(provider="openai")
        assert result == "gpt-4.1-nano"  # Should use default

    def test_to_str_override_provider(self):
        """Test string conversion overriding the detected provider"""
        identity = LLMIdentity("gpt-4", provider="openai")

        # Override with different provider
        result = identity.to_str(provider="openai")
        assert result == "gpt-4"

    def test_identity_string_passthrough(self):
        """Test that identity strings are returned as-is when known"""
        identity = LLMIdentity("custom-model-name")

        result = identity.to_str()
        assert result == "custom-model-name"


class TestProviderGuessing:
    """Test provider guessing functionality"""

    def test_guess_openai_provider(self):
        """Test guessing OpenAI provider from gpt- prefix"""
        assert guess_provider("gpt-4") == "openai"
        assert guess_provider("gpt-3.5-turbo") == "openai"
        assert guess_provider("gpt-4o") == "openai"
        assert guess_provider("gpt-4-turbo") == "openai"

    def test_guess_unknown_provider(self):
        """Test guessing unknown providers returns None"""
        assert guess_provider("claude-3") is None
        assert guess_provider("llama-2") is None
        assert guess_provider("custom-model") is None
        assert guess_provider("gemini-pro") is None

    def test_guess_none_identity(self):
        """Test guessing with None identity"""
        assert guess_provider(None) is None

    def test_guess_empty_string(self):
        """Test guessing with empty string"""
        assert guess_provider("") is None

    def test_guess_case_sensitive(self):
        """Test that guessing is case sensitive"""
        assert guess_provider("GPT-4") is None  # Should not match uppercase
        assert guess_provider("Gpt-4") is None  # Should not match mixed case


class TestIdentityIntegration:
    """Test integration scenarios with LLMIdentity"""

    def test_openai_workflow(self):
        """Test typical OpenAI model workflow"""
        # Create identity with OpenAI model
        identity = LLMIdentity("gpt-4")

        # Should auto-detect provider
        assert identity.provider == "openai"

        # Should return model name for OpenAI
        assert identity.to_str("openai") == "gpt-4"

        # Should work without explicit provider
        assert identity.to_str() == "gpt-4"

    def test_unknown_model_workflow(self):
        """Test workflow with unknown model"""
        # Create identity with unknown model
        identity = LLMIdentity("custom-model")

        # Should not detect provider
        assert identity.provider is None

        # Should still return model name
        assert identity.to_str() == "custom-model"

    def test_explicit_provider_override(self):
        """Test explicitly setting provider overrides guessing"""
        # Set explicit provider that doesn't match guessing
        identity = LLMIdentity("gpt-4", provider="custom")

        # Should use explicit provider, not guessed
        assert identity.provider == "custom"

        # Should still return model name
        assert identity.to_str() == "gpt-4"

    def test_default_model_behavior(self):
        """Test default model selection behavior"""
        # Create identity and clear the identity field
        identity = LLMIdentity("gpt-4")
        identity.identity = None

        # Should return default for OpenAI
        assert identity.to_str("openai") == "gpt-4.1-nano"

        # Should return None for unknown provider with None identity
        identity.provider = "unknown"
        assert identity.to_str() is None


class TestIdentityEdgeCases:
    """Test edge cases and error conditions"""

    def test_none_provider_explicit(self):
        """Test explicitly setting provider to None"""
        identity = LLMIdentity("gpt-4", provider=None)

        # Should still guess provider even when explicitly set to None
        assert identity.provider == "openai"

    def test_to_str_with_none_provider_parameter(self):
        """Test to_str with None provider parameter"""
        identity = LLMIdentity("test-model", provider="openai")

        # Should use instance provider when parameter is None
        result = identity.to_str(provider=None)
        assert result == "test-model"
