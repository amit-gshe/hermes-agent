"""Regression guard: don't send Anthropic ``thinking`` to Kimi's /coding endpoint.

Kimi's ``api.kimi.com/coding`` endpoint speaks the Anthropic Messages protocol
but has its own thinking semantics.  When ``thinking.enabled`` is present in
the request, Kimi validates the message history and requires every prior
assistant tool-call message to carry OpenAI-style ``reasoning_content``.

The Anthropic path never populates that field, and
``convert_messages_to_anthropic`` strips Anthropic thinking blocks on
third-party endpoints — so after one turn with tool calls the next request
fails with HTTP 400::

    thinking is enabled but reasoning_content is missing in assistant
    tool call message at index N

Kimi on the chat_completions route handles ``thinking`` via ``extra_body`` in
``ChatCompletionsTransport`` (#13503).  On the Anthropic route the right
thing to do is drop the parameter entirely and let Kimi drive reasoning
server-side.
"""

from __future__ import annotations

import pytest


class TestKimiCodingSkipsAnthropicThinking:
    """build_anthropic_kwargs must not inject ``thinking`` for Kimi /coding."""

    @pytest.mark.parametrize(
        "base_url",
        [
            "https://api.kimi.com/coding",
            "https://api.kimi.com/coding/v1",
            "https://api.kimi.com/coding/anthropic",
            "https://api.kimi.com/coding/",
        ],
    )
    def test_kimi_coding_endpoint_omits_thinking(self, base_url: str) -> None:
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="kimi-k2.5",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url=base_url,
        )
        assert "thinking" not in kwargs, (
            "Anthropic thinking must not be sent to Kimi /coding — "
            "endpoint requires reasoning_content on history we don't preserve."
        )
        assert "output_config" not in kwargs

    def test_kimi_coding_with_explicit_disabled_also_omits(self) -> None:
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="kimi-k2.5",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": False},
            base_url="https://api.kimi.com/coding",
        )
        assert "thinking" not in kwargs

    def test_non_kimi_third_party_still_gets_thinking(self) -> None:
        """MiniMax and other third-party Anthropic endpoints must retain thinking."""
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url="https://api.minimax.io/anthropic",
        )
        assert "thinking" in kwargs
        assert kwargs["thinking"]["type"] == "enabled"

    def test_native_anthropic_still_gets_thinking(self) -> None:
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url=None,
        )
        assert "thinking" in kwargs

    def test_kimi_root_endpoint_via_anthropic_transport_omits_thinking(self) -> None:
        """Plain ``api.kimi.com`` hit via the Anthropic transport also omits thinking.

        Auto-detection routes ``api.kimi.com/v1`` to ``chat_completions`` by
        default, but users can explicitly configure
        ``api_mode: anthropic_messages`` against any Kimi host.  The upstream
        validation (reasoning_content required on replayed tool-call
        messages) is the same regardless of URL path, so the thinking
        suppression must apply to every Kimi host, not just ``/coding``.
        See #17057.
        """
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="kimi-k2.5",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url="https://api.kimi.com/v1",
        )
        assert "thinking" not in kwargs

    # ── Official Kimi/Moonshot hosts (non-/coding path) ──────────
    @pytest.mark.parametrize(
        "base_url,model",
        [
            # Official Moonshot host (previously uncovered)
            ("https://api.moonshot.ai/anthropic", "moonshot-v1-32k"),
            ("https://api.moonshot.cn/anthropic", "moonshot-v1-32k"),
            # Official Kimi host (non-/coding path)
            ("https://api.kimi.com/v1", "kimi-k2.5"),
        ],
    )
    def test_official_kimi_moonshot_endpoint_omits_thinking(
        self, base_url: str, model: str
    ) -> None:
        """Official Kimi/Moonshot hosts must strip Anthropic thinking.

        Auto-detection routes ``api.kimi.com/v1`` to ``chat_completions`` by
        default, but users can explicitly configure
        ``api_mode: anthropic_messages`` against any Kimi/Moonshot host.
        The upstream validation (reasoning_content required on replayed
        tool-call messages) applies to every Kimi/Moonshot official host,
        not just ``/coding``.
        """
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url=base_url,
        )
        assert "thinking" not in kwargs, (
            f"Official Kimi/Moonshot endpoint ({base_url}, {model}) must not receive "
            f"Anthropic thinking — upstream validates reasoning_content on "
            f"replayed tool-call history we don't preserve."
        )
        assert "output_config" not in kwargs

    # ── Custom endpoints must keep thinking (model name alone is insufficient) ──────────
    @pytest.mark.parametrize(
        "base_url,model",
        [
            # Custom host with Kimi-family model — model name alone should NOT trigger suppression
            # This was the bug: custom gateways were misidentified as Kimi endpoints
            # based solely on model name, causing thinking to be skipped.
            # See user report: https://hub.test.utown.io + kimi-k2.6 → unable to think.
            ("http://my-kimi-proxy.internal", "kimi-2.6"),
            ("https://llm.example.com/anthropic", "kimi-k2.5"),
            ("https://llm.example.com/anthropic", "moonshot-v1-8k"),
            ("https://llm.example.com/anthropic", "kimi_thinking"),
            ("https://llm.example.com/anthropic", "moonshotai/kimi-k2.5"),
            # Custom endpoint with no base_url similarity to Kimi official domains
            ("https://hub.test.utown.io", "kimi-k2.6"),
        ],
    )
    def test_custom_endpoint_kimi_model_keeps_thinking(
        self, base_url: str, model: str
    ) -> None:
        """Custom endpoints with Kimi-family model names must keep thinking.

        Model-name-based detection was removed because it caused false positives:
        custom gateways using model names that happen to match Kimi prefixes
        were incorrectly treated as Kimi endpoints, causing thinking to be
        suppressed when the backend actually supported it.

        Users who genuinely proxy Kimi should use Kimi's official domain
        or configure their gateway to appear as api.kimi.com/moonshot.ai.
        """
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url=base_url,
        )
        assert "thinking" in kwargs, (
            f"Custom endpoint ({base_url}, {model}) must receive thinking — "
            f"model name alone is insufficient to identify as Kimi endpoint."
        )

    def test_custom_endpoint_non_kimi_model_keeps_thinking(self) -> None:
        """Custom endpoint with a non-Kimi model must keep thinking intact.

        Guards against over-broad model-family matching — only model names
        starting with a Kimi/Moonshot prefix should trigger suppression.
        """
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url="https://my-llm-proxy.example.com/anthropic",
        )
        assert "thinking" in kwargs
        assert kwargs["thinking"]["type"] == "enabled"

    def test_official_kimi_replay_preserves_unsigned_thinking(self) -> None:
        """On an official Kimi endpoint, unsigned reasoning_content thinking
        blocks must survive the third-party signature-stripping pass so
        the upstream's message-history validation passes.
        """
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "reasoning_content": "planning the tool call",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "skill_view", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ]
        _, converted = convert_messages_to_anthropic(
            messages,
            base_url="https://api.kimi.com/v1",
            model="kimi-2.6",
        )
        # The assistant message still carries the unsigned thinking block
        # synthesised from reasoning_content (required by Kimi's history
        # validation).  A plain third-party endpoint would have stripped it.
        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        assistant_blocks = assistant_msg["content"]
        thinking_blocks = [
            b for b in assistant_blocks
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "planning the tool call"

    def test_custom_endpoint_replay_strips_unsigned_thinking(self) -> None:
        """On a custom endpoint (not official Kimi), unsigned reasoning_content
        thinking blocks should be stripped because the endpoint is not
        identified as Kimi-based solely on model name.
        """
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "reasoning_content": "planning the tool call",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "skill_view", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ]
        _, converted = convert_messages_to_anthropic(
            messages,
            base_url="http://my-custom-proxy.internal",
            model="kimi-2.6",
        )
        # Custom endpoint is NOT treated as Kimi, so unsigned thinking blocks
        # from reasoning_content should be stripped (converted to text or removed).
        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        assistant_blocks = assistant_msg["content"]
        thinking_blocks = [
            b for b in assistant_blocks
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        # Unsigned thinking blocks should not survive on non-Kimi endpoints
        assert len(thinking_blocks) == 0
