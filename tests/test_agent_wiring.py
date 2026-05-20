"""Smoke tests for the supervisor agent wiring.

These tests do not call the LLM or Genie — they verify the graph is built
correctly and that the OBO contract is enforced.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_domains_have_required_fields():
    from agent_server.agent import DOMAINS

    assert len(DOMAINS) >= 1
    for d in DOMAINS:
        assert {"name", "space_id_env", "tool_description", "system_prompt"} <= d.keys()
        assert d["name"]
        assert d["space_id_env"].endswith("_SPACE_ID")
        assert d["tool_description"]
        assert d["system_prompt"]


def test_prompts_forbid_fallback_to_memory():
    from agent_server import prompts

    # L1 must delegate, never answer directly.
    assert "delegate" in prompts.L1_ROUTER.lower() or "do not answer" in prompts.L1_ROUTER.lower()

    # Each L2 must require a Genie call and must NOT paper over permission errors.
    for body in (prompts.FINANCE_L2, prompts.SALES_L2):
        assert "genie" in body.lower()
        assert "permission" in body.lower()


def test_obo_fallback_disabled_by_default(monkeypatch):
    """Without the env flag and without the user token, OBO must fail loudly."""
    from agent_server.utils import get_user_workspace_client

    monkeypatch.delenv("OBO_FALLBACK_TO_DEFAULT", raising=False)
    with patch("agent_server.utils.get_request_headers", return_value={}):
        with pytest.raises(RuntimeError, match="x-forwarded-access-token"):
            get_user_workspace_client()


def test_obo_fallback_to_default_when_flagged(monkeypatch):
    from agent_server.utils import get_user_workspace_client

    monkeypatch.setenv("OBO_FALLBACK_TO_DEFAULT", "1")
    sentinel = object()
    with (
        patch("agent_server.utils.get_request_headers", return_value={}),
        patch("agent_server.utils.WorkspaceClient", return_value=sentinel) as ws,
    ):
        client = get_user_workspace_client()
        assert client is sentinel
        ws.assert_called_once_with()  # no token arg => default CLI auth


def test_obo_uses_forwarded_token():
    from agent_server.utils import get_user_workspace_client

    headers = {"x-forwarded-access-token": "user-token-xyz"}
    with (
        patch("agent_server.utils.get_request_headers", return_value=headers),
        patch("agent_server.utils.WorkspaceClient") as ws,
    ):
        get_user_workspace_client()
        ws.assert_called_once_with(token="user-token-xyz", auth_type="pat")


def test_build_l1_agent_constructs_one_tool_per_domain(monkeypatch):
    """build_l1_agent should produce N L2 handoffs (one per DOMAIN)."""
    monkeypatch.setenv("GENIE_FINANCE_SPACE_ID", "fin-space-id")
    monkeypatch.setenv("GENIE_SALES_SPACE_ID", "sales-space-id")

    from agent_server import agent as agent_mod

    # Stub the L2 builder and the L1 create_agent so we don't hit MCP or the LLM.
    fake_l2 = MagicMock()
    fake_l2.ainvoke = AsyncMock(return_value={"messages": [MagicMock(content="ok")]})

    async def fake_build_l2(_user_ws, _domain):
        return fake_l2

    captured = {}

    def fake_create_agent(tools, model, prompt):  # noqa: ARG001
        captured["tools"] = tools
        captured["prompt"] = prompt
        return MagicMock()

    monkeypatch.setattr(agent_mod, "_build_l2_supervisor", fake_build_l2)
    monkeypatch.setattr(agent_mod, "create_agent", fake_create_agent)
    monkeypatch.setattr(agent_mod, "ChatDatabricks", lambda endpoint: MagicMock(endpoint=endpoint))

    import asyncio

    asyncio.run(agent_mod.build_l1_agent(user_ws=MagicMock()))

    assert captured["prompt"] == agent_mod.prompts.L1_ROUTER
    assert len(captured["tools"]) == len(agent_mod.DOMAINS)
    tool_names = {t.name for t in captured["tools"]}
    assert tool_names == {f"ask_{d['name']}" for d in agent_mod.DOMAINS}


def test_build_l2_supervisor_requires_space_id_env(monkeypatch):
    """If the Genie space env var is missing, building L2 must raise."""
    monkeypatch.delenv("GENIE_FINANCE_SPACE_ID", raising=False)
    from agent_server import agent as agent_mod

    finance_domain = next(d for d in agent_mod.DOMAINS if d["name"] == "finance")

    import asyncio

    with pytest.raises(RuntimeError, match="GENIE_FINANCE_SPACE_ID"):
        asyncio.run(agent_mod._build_l2_supervisor(user_ws=MagicMock(), domain=finance_domain))
