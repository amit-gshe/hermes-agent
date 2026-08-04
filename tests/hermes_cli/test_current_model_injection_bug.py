"""Regression test for current_model injection when live discovery replaces config models.

Scenario (from user report):
- User has a custom provider in providers: with api_mode: anthropic_messages
- model.default is set to a model that is NOT in the live /models catalog
- Live discovery replaces the explicit models list from config
- The current_model injection post-pass should still show the configured default model

Root cause hypothesis:
1. api_mode is not passed to fetch_api_models() → wrong auth header (Bearer instead of x-api-key)
2. But even if auth succeeds and returns a different list, current_model injection should still work
3. The injection fails if is_current is False (slug mismatch) or if the model is already present
"""

import pytest
from unittest.mock import patch, MagicMock

from hermes_cli.model_switch import list_authenticated_providers


def test_current_model_injection_when_live_discovery_replaces_config_models():
    """When live discovery returns a list without current_model, injection should add it.

    This tests the scenario where:
    - providers: has a custom provider with api_mode: anthropic_messages
    - model.default is 'deepseek-v4-flash'
    - Live discovery returns a list that does NOT include 'deepseek-v4-flash'
    - The current_model injection should still add it to the picker
    """
    # Mock models_dev to avoid network calls
    with patch("agent.models_dev.fetch_models_dev", return_value={}):
        # Mock the live discovery to return a list without our target model
        mock_fetch = MagicMock(return_value=[
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ])

        with patch("hermes_cli.models.fetch_api_models", mock_fetch):
            providers = list_authenticated_providers(
                current_provider="claude-code-hub",
                current_model="deepseek-v4-flash",
                user_providers={
                    "claude-code-hub": {
                        "name": "Claude Code Hub",
                        "base_url": "https://api.example.com/v1",
                        "api_key": "sk-test-key",
                        "api_mode": "anthropic_messages",
                        "models": ["deepseek-v4-flash", "claude-3-5-sonnet-20241022"],
                    }
                },
                custom_providers=[],
                probe_custom_providers=True,
            )

    # Find our provider row
    hub_row = next((p for p in providers if p["slug"] == "claude-code-hub"), None)
    assert hub_row is not None, "Provider row should exist"
    assert hub_row["is_current"] is True, "Provider should be marked as current"

    # The bug: current_model is missing from the models list
    models = hub_row.get("models", [])
    print(f"\nModels in picker row: {models}")
    print(f"Current model: deepseek-v4-flash")
    print(f"Is current: {hub_row['is_current']}")

    # This assertion should FAIL if the bug exists
    assert "deepseek-v4-flash" in models, (
        f"current_model 'deepseek-v4-flash' should be injected into models list. "
        f"Got: {models}"
    )


def test_api_mode_passed_to_fetch_api_models_for_anthropic():
    """Verify that api_mode is forwarded to fetch_api_models for auth header selection.

    When a provider has api_mode: anthropic_messages, the probe should use
    x-api-key header instead of Authorization: Bearer.
    """
    with patch("agent.models_dev.fetch_models_dev", return_value={}):
        mock_fetch = MagicMock(return_value=["model-a", "model-b"])

        with patch("hermes_cli.models.fetch_api_models", mock_fetch):
            providers = list_authenticated_providers(
                current_provider="test-provider",
                current_model="model-a",
                user_providers={
                    "test-provider": {
                        "name": "Test Provider",
                        "base_url": "https://api.test.com/v1",
                        "api_key": "test-key",
                        "api_mode": "anthropic_messages",
                    }
                },
                custom_providers=[],
                probe_custom_providers=True,
            )

    # Verify fetch_api_models was called
    assert mock_fetch.called, "fetch_api_models should be called"

    # Check the call arguments
    call_args = mock_fetch.call_args
    print(f"\nfetch_api_models called with:")
    print(f"  api_key: {call_args[0][0] if call_args[0] else call_args[1].get('api_key')}")
    print(f"  base_url: {call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get('base_url')}")
    print(f"  api_mode: {call_args[1].get('api_mode') if call_args[1] else 'NOT PASSED'}")

    # This is the BUG: api_mode is not passed
    # The call should pass api_mode="anthropic_messages"
    if call_args[1]:
        assert call_args[1].get("api_mode") == "anthropic_messages", (
            "api_mode should be forwarded to fetch_api_models"
        )
    else:
        # If called with positional args only, api_mode won't be passed
        pytest.fail("api_mode was not passed to fetch_api_models (this is the bug)")


def test_current_model_injection_is_case_sensitive():
    """Verify that current_model injection is case-sensitive.

    If live discovery returns 'DeepSeek-V4-Flash' but config has 'deepseek-v4-flash',
    the injection should still add the configured version.
    """
    with patch("agent.models_dev.fetch_models_dev", return_value={}):
        mock_fetch = MagicMock(return_value=[
            "DeepSeek-V4-Flash",  # Different case
            "Claude-3-5-Sonnet",
        ])

        with patch("hermes_cli.models.fetch_api_models", mock_fetch):
            providers = list_authenticated_providers(
                current_provider="test-provider",
                current_model="deepseek-v4-flash",  # Lowercase
                user_providers={
                    "test-provider": {
                        "name": "Test Provider",
                        "base_url": "https://api.test.com/v1",
                        "api_key": "test-key",
                    }
                },
                custom_providers=[],
                probe_custom_providers=True,
            )

    hub_row = next((p for p in providers if p["slug"] == "test-provider"), None)
    assert hub_row is not None

    models = hub_row.get("models", [])
    print(f"\nModels: {models}")

    # Both versions should be present (different case = different strings)
    # The injection adds the lowercase version since it's not in the list
    assert "deepseek-v4-flash" in models, (
        "Configured lowercase model should be injected even if uppercase version exists"
    )


def test_current_model_injection_with_empty_live_discovery():
    """When live discovery returns empty list, explicit models should be preserved."""
    with patch("agent.models_dev.fetch_models_dev", return_value={}):
        # Live discovery returns None (failure)
        mock_fetch = MagicMock(return_value=None)

        with patch("hermes_cli.models.fetch_api_models", mock_fetch):
            providers = list_authenticated_providers(
                current_provider="test-provider",
                current_model="deepseek-v4-flash",
                user_providers={
                    "test-provider": {
                        "name": "Test Provider",
                        "base_url": "https://api.test.com/v1",
                        "api_key": "test-key",
                        "models": ["deepseek-v4-flash", "other-model"],
                    }
                },
                custom_providers=[],
                probe_custom_providers=True,
            )

    hub_row = next((p for p in providers if p["slug"] == "test-provider"), None)
    assert hub_row is not None
    models = hub_row.get("models", [])
    print(f"\nModels when live discovery fails: {models}")

    # Explicit models should be preserved when live discovery fails
    assert "deepseek-v4-flash" in models
    assert "other-model" in models


def test_is_current_flag_is_set_correctly():
    """Verify that is_current flag is set correctly based on provider name match."""
    with patch("agent.models_dev.fetch_models_dev", return_value={}):
        mock_fetch = MagicMock(return_value=["model-a"])

        with patch("hermes_cli.models.fetch_api_models", mock_fetch):
            providers = list_authenticated_providers(
                current_provider="my-provider",
                current_model="model-x",
                user_providers={
                    "my-provider": {
                        "name": "My Provider",
                        "base_url": "https://api.test.com/v1",
                        "api_key": "test-key",
                    },
                    "other-provider": {
                        "name": "Other Provider",
                        "base_url": "https://api.other.com/v1",
                        "api_key": "test-key",
                    }
                },
                custom_providers=[],
                probe_custom_providers=True,
            )

    my_row = next((p for p in providers if p["slug"] == "my-provider"), None)
    other_row = next((p for p in providers if p["slug"] == "other-provider"), None)

    assert my_row is not None and other_row is not None
    print(f"\nmy-provider is_current: {my_row['is_current']}")
    print(f"other-provider is_current: {other_row['is_current']}")

    assert my_row["is_current"] is True, "Current provider should have is_current=True"
    assert other_row["is_current"] is False, "Non-current provider should have is_current=False"

    # current_model should be injected into the current provider only
    assert "model-x" in my_row.get("models", [])
    assert "model-x" not in other_row.get("models", [])


def test_injection_with_case_mismatch_in_provider_name():
    """Test what happens when provider name has case mismatch.

    The is_current check is: ep_name == current_provider
    This is case-sensitive! If config has 'My-Provider' but current_provider
    is 'my-provider', is_current will be False.
    """
    with patch("agent.models_dev.fetch_models_dev", return_value={}):
        mock_fetch = MagicMock(return_value=["model-a"])

        with patch("hermes_cli.models.fetch_api_models", mock_fetch):
            providers = list_authenticated_providers(
                current_provider="my-provider",  # lowercase
                current_model="model-x",
                user_providers={
                    "My-Provider": {  # Different case!
                        "name": "My Provider",
                        "base_url": "https://api.test.com/v1",
                        "api_key": "test-key",
                    },
                },
                custom_providers=[],
                probe_custom_providers=True,
            )

    # The slug comes from the dict key (My-Provider), not from current_provider
    hub_row = next((p for p in providers if p["slug"] == "My-Provider"), None)
    assert hub_row is not None

    print(f"\nProvider slug: {hub_row['slug']}")
    print(f"current_provider: my-provider")
    print(f"is_current: {hub_row['is_current']}")

    # This is a potential bug: case mismatch causes is_current to be False
    # and current_model is NOT injected!
    if hub_row["is_current"] is False:
        print("WARNING: Case mismatch caused is_current to be False!")
        print("This means current_model will NOT be injected!")
        # This is the bug!
        pytest.fail(
            "Case mismatch in provider name caused is_current=False, "
            "preventing current_model injection"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
