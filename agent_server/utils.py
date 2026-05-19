import logging
import os
from typing import Any, AsyncGenerator, AsyncIterator, Optional

from databricks.sdk import WorkspaceClient
from databricks_langchain.chat_models import json
from langchain.messages import AIMessageChunk, ToolMessage
from mlflow.genai.agent_server import get_request_headers
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentStreamEvent,
    create_text_delta,
    output_to_responses_items_stream,
)


def get_session_id(request: ResponsesAgentRequest) -> str | None:
    if request.context and request.context.conversation_id:
        return request.context.conversation_id
    if request.custom_inputs and isinstance(request.custom_inputs, dict):
        return request.custom_inputs.get("session_id")
    return None


def get_user_workspace_client() -> WorkspaceClient:
    """Return a WorkspaceClient that acts on behalf of the calling user.

    On Databricks Apps, the platform injects `x-forwarded-access-token` with a
    downscoped user token (see `user_api_scopes` in `manifest.yaml`).

    Locally, that header is absent. When `OBO_FALLBACK_TO_DEFAULT=1` we fall
    back to the developer's CLI profile so the dev loop still works; this
    flag MUST be unset in eval/prod, otherwise the agent would silently run
    Genie queries with the developer's identity.
    """
    headers = get_request_headers() or {}
    token = headers.get("x-forwarded-access-token")
    if token:
        return WorkspaceClient(token=token, auth_type="pat")
    if os.environ.get("OBO_FALLBACK_TO_DEFAULT") == "1":
        logging.warning(
            "No x-forwarded-access-token; falling back to default CLI auth "
            "(OBO_FALLBACK_TO_DEFAULT=1). DO NOT enable this in prod."
        )
        return WorkspaceClient()
    raise RuntimeError(
        "No x-forwarded-access-token header found. The agent requires an "
        "end-user token to call Genie on behalf of the user. For local dev, "
        "set OBO_FALLBACK_TO_DEFAULT=1."
    )


def get_databricks_host_from_env() -> Optional[str]:
    try:
        w = WorkspaceClient()
        return w.config.host
    except Exception as e:
        logging.exception(f"Error getting databricks host from env: {e}")
        return None


async def process_agent_astream_events(
    async_stream: AsyncIterator[Any],
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    """
    Generic helper to process agent stream events and yield ResponsesAgentStreamEvent objects.

    Args:
        async_stream: The async iterator from agent.astream()
    """
    async for event in async_stream:
        if event[0] == "updates":
            for node_data in event[1].values():
                if len(node_data.get("messages", [])) > 0:
                    for msg in node_data["messages"]:
                        if isinstance(msg, ToolMessage) and not isinstance(msg.content, str):
                            msg.content = json.dumps(msg.content)
                    for item in output_to_responses_items_stream(node_data["messages"]):
                        yield item
        elif event[0] == "messages":
            try:
                chunk = event[1][0]
                if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                    yield ResponsesAgentStreamEvent(
                        **create_text_delta(delta=content, item_id=chunk.id)
                    )
            except Exception as e:
                logging.exception(f"Error processing agent stream event: {e}")
