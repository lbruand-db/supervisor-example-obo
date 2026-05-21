"""Hierarchical supervisor agent: L1 router -> L2 domain supervisor -> Genie space (OBO).

LLM and infra calls run under the app's service principal. Genie tool calls
run under the end-user's identity (`x-forwarded-access-token`), so Unity
Catalog grants on the underlying tables are enforced per caller.
"""

import logging
import os
from collections.abc import AsyncGenerator

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent
from langchain_core.tools import tool
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)

from agent_server import prompts
from agent_server.utils import (
    get_session_id,
    get_user_workspace_client,
    process_agent_astream_events,
)

logger = logging.getLogger(__name__)
mlflow.langchain.autolog()
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)

LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "databricks-gpt-5-2")


# Edit this list (or extend it via env-driven config) to add domains. Each
# entry produces one L2 supervisor with its own Genie space, exposed to L1 as
# a tool named `ask_<name>`.
DOMAINS = [
    {
        "name": "finance",
        "space_id_env": "GENIE_FINANCE_SPACE_ID",
        "tool_description": (
            "Answer finance, accounting, revenue, and cost questions by querying "
            "the finance Genie space. Use for KPIs like YTD revenue, margin, opex."
        ),
        "system_prompt": prompts.FINANCE_L2,
    },
    {
        "name": "sales",
        "space_id_env": "GENIE_SALES_SPACE_ID",
        "tool_description": (
            "Answer sales pipeline, opportunity, quota, and bookings questions by "
            "querying the sales Genie space. Use for pipeline coverage, win rate, ARR."
        ),
        "system_prompt": prompts.SALES_L2,
    },
]


def _build_genie_tool(user_ws: WorkspaceClient, name: str, space_id: str):
    """A single LangChain tool that asks the Genie space `space_id` a question.

    Uses the direct Genie REST API via the SDK (start_conversation_and_wait)
    bound to `user_ws`. This runs under the end-user's identity and Unity
    Catalog grants — the calling user must have CAN_RUN on the Genie space
    AND grants on the underlying tables.

    Why not the MCP route (/api/2.0/mcp/genie/{space_id}): that endpoint
    requires a scope beyond the documented user_api_scopes allowlist
    (`sql`, `dashboards.genie`, `files.files`), so it 403s under OBO. The
    direct Genie API works with just `dashboards.genie`.
    """

    @tool(
        name_or_callable=f"genie_{name}",
        description=(
            f"Query the {name} Genie space with a natural-language question. "
            "Returns Genie's textual answer plus any SQL it generated. "
            "Forwards permission errors verbatim — do not retry under another identity."
        ),
    )
    def _query(question: str) -> str:
        msg = user_ws.genie.start_conversation_and_wait(space_id, question)
        parts: list[str] = []
        for att in msg.attachments or []:
            if att.text and att.text.content:
                parts.append(att.text.content)
            if att.query:
                if att.query.description:
                    parts.append(f"_{att.query.description}_")
                if att.query.query:
                    parts.append(f"```sql\n{att.query.query}\n```")
        if not parts and msg.content:
            parts.append(msg.content)
        return "\n\n".join(parts) if parts else "(no response from Genie)"

    return _query


def _build_l2_supervisor(user_ws: WorkspaceClient, domain: dict):
    space_id = os.environ.get(domain["space_id_env"])
    if not space_id:
        raise RuntimeError(
            f"Missing env {domain['space_id_env']} for domain '{domain['name']}'. "
            "Set it in app.yaml / .env."
        )
    genie_tool = _build_genie_tool(user_ws, domain["name"], space_id)
    return create_agent(
        tools=[genie_tool],
        model=ChatDatabricks(endpoint=LLM_ENDPOINT),
        system_prompt=domain["system_prompt"],
    )


def _make_l1_tool(name: str, description: str, child_agent):
    """Wrap an L2 supervisor as a single tool the L1 router can call."""

    @tool(name_or_callable=f"ask_{name}", description=description)
    async def _ask(question: str) -> str:
        result = await child_agent.ainvoke({"messages": [("user", question)]})
        return result["messages"][-1].content

    return _ask


def build_l1_agent(user_ws: WorkspaceClient):
    """Build the full L1 -> L2 -> Genie graph for a single request."""
    handoff_tools = []
    for domain in DOMAINS:
        l2 = _build_l2_supervisor(user_ws, domain)
        handoff_tools.append(_make_l1_tool(domain["name"], domain["tool_description"], l2))

    return create_agent(
        tools=handoff_tools,
        model=ChatDatabricks(endpoint=LLM_ENDPOINT),
        system_prompt=prompts.L1_ROUTER,
    )


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    outputs = [
        event.item
        async for event in stream_handler(request)
        if event.type == "response.output_item.done"
    ]
    return ResponsesAgentResponse(output=outputs)


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    if session_id := get_session_id(request):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

    user_ws = get_user_workspace_client()
    agent = build_l1_agent(user_ws)

    messages = {"messages": to_chat_completions_input([i.model_dump() for i in request.input])}

    async for event in process_agent_astream_events(
        agent.astream(input=messages, stream_mode=["updates", "messages"])
    ):
        yield event
