"""One-screen MLflow ResponsesAgent that echoes every identity source it can
find inside a Mosaic AI serving endpoint, including the documented OBO
(on-behalf-of-user) path.

The agent reports, for each query:

  * Whatever ChatContext / custom_inputs the request carried.
  * The default `WorkspaceClient().current_user.me()` — expected to be the
    endpoint's service principal.
  * The OBO `WorkspaceClient(credentials_strategy=ModelServingUserCredentials())`
    — expected to be the calling user once `auth_policy=AuthPolicy(...)` has
    been attached at log time.
  * A live Genie call against a hard-coded space, run under each client,
    so we can prove which one actually traverses the user's UC grants.
"""

from __future__ import annotations

import json
import os

from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
)

# A Genie space the spike user owns (created earlier in this workspace).
# If you fork this for another workspace, point it at one of yours.
PROBE_GENIE_SPACE_ID = "01f15474886a16d5aa027e0791fa855a"

INTERESTING_ENV_KEYS = [
    "DATABRICKS_HOST",
    "DATABRICKS_WORKSPACE_ID",
    "DATABRICKS_CLIENT_ID",
    "DB_MODEL_SERVING_HOST_URL",
    "DB_INSTANCE_ID",
    "DATABRICKS_AGENTS_AUTH_TOKEN",
    "DATABRICKS_USER_TOKEN",
    "DATABRICKS_REQUEST_USER",
    "INVOKER_USER_NAME",
]


def _safe(callable_, default="<error>"):
    try:
        return callable_()
    except Exception as e:
        return f"{default} ({type(e).__name__}: {e})"


def _ws_identity(ws) -> dict:
    me = ws.current_user.me()
    return {
        "host": ws.config.host,
        "auth_type": ws.config.auth_type,
        "user_name": me.user_name,
        "display_name": me.display_name,
    }


def _genie_smoke(ws, space_id: str) -> dict:
    """Send a trivial question to a Genie space using the given client.

    We don't care about the answer — only whether the call is allowed.
    A 403 / permission error here for the OBO client (but not the SP
    client) is exactly the signal that proves OBO is intact.
    """
    try:
        msg = ws.genie.start_conversation_and_wait(space_id, "Count rows in any table.")
        # If we got here at all, the user / SP was allowed to talk to the space.
        return {
            "status": "ok",
            "message_status": str(getattr(msg, "status", None)),
            "has_attachments": bool(getattr(msg, "attachments", None)),
        }
    except Exception as e:
        return {"status": "error", "type": type(e).__name__, "message": str(e)[:400]}


def collect_identity_report(request: ResponsesAgentRequest) -> dict:
    report: dict = {}

    if request.context is not None:
        report["context"] = {
            "user_id": request.context.user_id,
            "conversation_id": request.context.conversation_id,
        }
    else:
        report["context"] = None

    report["custom_inputs"] = request.custom_inputs

    env_view = {}
    for k in INTERESTING_ENV_KEYS:
        v = os.environ.get(k)
        if v is None:
            env_view[k] = None
        elif len(v) > 24:
            env_view[k] = f"{v[:6]}…{v[-4:]} (len={len(v)})"
        else:
            env_view[k] = v
    report["env"] = env_view

    # Default SP-bound client.
    def _default_ws():
        from databricks.sdk import WorkspaceClient

        return _ws_identity(WorkspaceClient())

    report["workspace_client_default"] = _safe(_default_ws)

    # OBO-bound client — only meaningful once auth_policy was attached at
    # log time. Inside __init__ it raises; here in predict() it's valid.
    def _obo_ws_identity():
        from databricks.sdk import WorkspaceClient
        from databricks_ai_bridge import ModelServingUserCredentials

        ws = WorkspaceClient(credentials_strategy=ModelServingUserCredentials())
        return _ws_identity(ws)

    report["workspace_client_obo"] = _safe(_obo_ws_identity)

    # Live Genie call to prove which identity actually traverses UC.
    def _genie_default():
        from databricks.sdk import WorkspaceClient

        return _genie_smoke(WorkspaceClient(), PROBE_GENIE_SPACE_ID)

    def _genie_obo():
        from databricks.sdk import WorkspaceClient
        from databricks_ai_bridge import ModelServingUserCredentials

        ws = WorkspaceClient(credentials_strategy=ModelServingUserCredentials())
        return _genie_smoke(ws, PROBE_GENIE_SPACE_ID)

    report["genie_default"] = _safe(_genie_default)
    report["genie_obo"] = _safe(_genie_obo)
    report["probe_space_id"] = PROBE_GENIE_SPACE_ID

    return report


class IdentityEchoAgent(ResponsesAgent):
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        report = collect_identity_report(request)
        body = "## Identity report\n```json\n" + json.dumps(report, indent=2, default=str) + "\n```"
        item = self.create_text_output_item(text=body, id="msg-identity")
        return ResponsesAgentResponse(output=[item])


from mlflow.models import set_model  # noqa: E402

set_model(IdentityEchoAgent())
