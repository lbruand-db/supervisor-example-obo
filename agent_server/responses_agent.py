"""MLflow ResponsesAgent wrapper around the production L1→L2→Genie graph.

This is the entry point logged into MLflow and served by Mosaic AI Model
Serving. It is NOT used by the Databricks Apps build (the Apps build
runs `agent_server/agent.py`'s `invoke_handler` / `stream_handler` via
FastAPI / `mlflow.genai.agent_server` decorators).

Identity binding: see `SPECS/PLAN_MODEL_SERVING.md` §6 for the full
story. Short version:

  * On Apps, `x-forwarded-access-token` is forwarded by the platform
    proxy and lifted by `utils.get_user_workspace_client`.
  * On Model Serving, the equivalent is
    `WorkspaceClient(credentials_strategy=ModelServingUserCredentials())`,
    which MUST be instantiated inside `predict()` / `predict_stream()`.
    Both require the workspace "Agent Framework: On-Behalf-Of-User
    Authorization" preview to be enabled, AND the endpoint must be
    (re)deployed after enabling.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from typing import Any

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_ai_bridge import ModelServingUserCredentials
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)

from agent_server.agent import build_l1_agent
from agent_server.utils import process_agent_astream_events

mlflow.langchain.autolog()


class SupervisorAgent(ResponsesAgent):
    """ResponsesAgent shell that delegates to the production supervisor graph."""

    async def _astream(self, request: ResponsesAgentRequest) -> Any:
        # `ModelServingUserCredentials` only resolves when invoked inside a
        # live serving request. Building the workspace client (and therefore
        # the agent) here, per-request, is required.
        user_ws = WorkspaceClient(credentials_strategy=ModelServingUserCredentials())
        agent = build_l1_agent(user_ws)
        messages = {"messages": to_chat_completions_input([i.model_dump() for i in request.input])}
        async for event in process_agent_astream_events(
            agent.astream(input=messages, stream_mode=["updates", "messages"])
        ):
            yield event

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        async def _collect():
            outputs = []
            async for event in self._astream(request):
                if event.type == "response.output_item.done":
                    outputs.append(event.item)
            return outputs

        return ResponsesAgentResponse(output=asyncio.run(_collect()))

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        # MLflow's predict_stream contract is a sync generator. The
        # production graph is async, so we bridge: run one step of the
        # async generator on each iteration, yielding the result back to
        # the sync caller. This preserves the per-event streaming
        # behaviour rather than buffering everything first.
        loop = asyncio.new_event_loop()
        try:
            agen = self._astream(request)
            while True:
                try:
                    yield loop.run_until_complete(agen.__anext__())
                except StopAsyncIteration:
                    return
        finally:
            loop.close()


from mlflow.models import set_model  # noqa: E402

set_model(SupervisorAgent())
