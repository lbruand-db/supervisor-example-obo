"""Second spike: wrap the real L1→L2→Genie graph as a ResponsesAgent and
serve it from Mosaic AI Model Serving with OBO.

The first spike (`spikes/serving-obo/`) proved the OBO mechanism in
isolation. This one proves the actual production graph reuses cleanly:

  * Subclass `mlflow.pyfunc.ResponsesAgent`.
  * Build the user-bound `WorkspaceClient` inside `predict()` via
    `ModelServingUserCredentials` — same pattern as spike #1.
  * Call `agent_server.agent.build_l1_agent(user_ws)` (production code,
    unmodified) to get the LangGraph runnable.
  * Bridge async → sync by running `agent.ainvoke(...)` under asyncio.

If this works, PLAN_MODEL_SERVING.md §4.1 ("Reused as-is") is no
longer hand-waving.
"""

from __future__ import annotations

import asyncio

from databricks.sdk import WorkspaceClient
from databricks_ai_bridge import ModelServingUserCredentials
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
)

from agent_server.agent import build_l1_agent


class SupervisorAgent(ResponsesAgent):
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        # MUST be inside predict(): ModelServingUserCredentials needs the
        # caller's identity, which is only attached to this request's
        # execution context.
        user_ws = WorkspaceClient(credentials_strategy=ModelServingUserCredentials())
        agent = build_l1_agent(user_ws)

        # Convert ResponsesAgentRequest inputs to LangChain message tuples.
        # Each input item is one of OutputItem / Message / FunctionCall etc;
        # for a fresh chat, the caller sends [{role, content}] and the
        # dumps are simple. Keep this loose — pydantic objects also expose
        # .role and .content as attributes.
        lc_messages = []
        for item in request.input:
            role = getattr(item, "role", None) or item.get("role")
            content = getattr(item, "content", None) or item.get("content")
            # content can be a string OR a list of content parts; collapse
            # the latter into text.
            if isinstance(content, list):
                parts = []
                for c in content:
                    text = c.get("text") if isinstance(c, dict) else getattr(c, "text", None)
                    if text:
                        parts.append(text)
                content = "\n".join(parts)
            lc_messages.append((role or "user", content or ""))

        result = asyncio.run(agent.ainvoke({"messages": lc_messages}))

        # Final assistant message is the last entry in result["messages"].
        final = result["messages"][-1]
        text = getattr(final, "content", None) or ""
        if isinstance(text, list):
            text = "\n".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in text)

        item = self.create_text_output_item(text=text or "(empty)", id="msg-final")
        return ResponsesAgentResponse(output=[item])


from mlflow.models import set_model  # noqa: E402

set_model(SupervisorAgent())
