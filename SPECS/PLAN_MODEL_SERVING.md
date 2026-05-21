# PLAN — Port the supervisor agent to a Model Serving endpoint

This is a **plan**, not an implementation. It describes how to take the
current Databricks Apps deployment of the L1→L2→Genie supervisor and
re-deploy the same agent as a Mosaic AI **Model Serving endpoint** behind
an MLflow `ResponsesAgent` model. The branch that does this work should
land alongside the current Apps build (likely on a `serving/` branch or as
a second target in the bundle), not replace it.

Cross-references: [`SPEC.md`](SPEC.md) for the existing design; the rules
about OBO at the leaf (§5) and the documented scope allowlist (§6.1, §12)
still apply.

---

## 1. Goal

Expose the same `/responses`-compatible agent at
`https://<workspace>/serving-endpoints/supervisor-example-obo/invocations`
(OpenAI Responses-compatible), with:

- The same L1→L2→Genie graph code reused largely unchanged (only the
  identity-binding line moves).
- Per-caller identity preserved for Genie reads (so UC grants on the
  underlying tables still gate access). Spike-verified in §6.
- One bundle target (`-t serving`) that builds, logs, registers, and
  deploys the endpoint end-to-end via `databricks bundle deploy`.

Non-goal: deprecating the Apps deployment. Apps + Serving are
complementary; this plan exists so customers can pick whichever fits.

## 1.1 Prerequisites (READ THIS FIRST)

This port only works on a workspace where a specific public preview has
been turned on. Without it the OBO path returns a runtime `ValueError`
and every Genie call has to fall back to the endpoint's service
principal — which defeats the point. Confirm before doing any of the
implementation work.

| Prereq | Where | How |
|---|---|---|
| Public preview **"Agent Framework: On-Behalf-Of-User Authorization"** must be **enabled on the workspace** | Workspace Admin → Settings → Previews | UI-only toggle. Not exposed via `databricks settings` CLI or `/api/2.0/previews*`. Requires workspace-admin role. |
| Endpoint must be **(re)deployed after** the preview is on | `uv run deploy-serving` (a redeploy is enough; the model artifact stays the same) | The runtime checks the preview at deploy time, not request time. Endpoints logged *before* the toggle keep failing even once it's on; only endpoints deployed *after* the toggle pick it up. |
| Caller must hit the endpoint with an **OAuth user token** (U2M), not a PAT | `databricks auth login --host <ws>` | A PAT bound to the same user works for `/api/2.0/*` but not for the serving endpoint's per-caller propagation. |

Customers without workspace-admin rights to flip the preview should
stay on the Apps build — there's no equivalent prereq there.

## 2. Why bother — trade-offs vs the Apps deployment

| Concern | Databricks App (today) | Model Serving (this plan) |
|---|---|---|
| Where the agent loop runs | App process (uvicorn) | Serving endpoint runtime (managed) |
| Scaling | App compute size, manual | Workload size + scale-to-zero |
| Cold start | App stays warm if compute is up | Cold-start cost on scale-to-zero |
| Built-in UI | Chat UI ships in the app | AI Playground / Review App |
| Caller identity | `x-forwarded-access-token` forwarded as-is | Constrained — see §6 |
| Eval / MLflow lineage | App writes traces to experiment | First-class; endpoint version = model version |
| Packaging | `source_code_path: ./` | Logged MLflow model in UC |
| OpenAI-client interop | Yes (via app URL) | Yes (via endpoint URL, no Apps proxy hop) |
| Deployment surface | Apps OAuth + workspace | UC catalog grants + workspace |

Pick Serving when callers are **other services** or **agents**. Pick Apps
when callers are **humans in a browser**.

## 3. Target architecture

```mermaid
flowchart TD
    caller([Caller<br/>OpenAI SDK / curl / another agent])

    subgraph SP_BOX["Serving endpoint runtime — runs as endpoint's SP"]
        EP[/"Endpoint:<br/>supervisor-example-obo"/]
        MODEL["MLflow ResponsesAgent model<br/>(logged with langchain flavor)"]
        L1["L1 router supervisor"]
        L2F["L2 supervisor – Finance"]
        L2S["L2 supervisor – Sales"]
        LLM[/"Chat LLM endpoint<br/>databricks-gpt-5-2"/]
        EP --> MODEL
        MODEL --> L1
    end

    subgraph OBO_BOX["Per-caller identity context (see §6)"]
        GF[("Genie space – Finance")]
        GS[("Genie space – Sales")]
    end

    caller -- "POST /serving-endpoints/<name>/invocations" --> EP
    L1 -- "ask_finance" --> L2F
    L1 -- "ask_sales"   --> L2S
    L1 -. LLM call .-> LLM
    L2F -. answer synthesis .-> LLM
    L2S -. answer synthesis .-> LLM
    L2F ==> GF
    L2S ==> GS
```

## 4. What changes vs the Apps build

### 4.1 Reused as-is

- `agent_server/prompts.py` — system prompts.
- `agent_server/agent.py` — the `DOMAINS` list, `_build_genie_tool`,
  `_build_l2_supervisor`, `build_l1_agent`, `_make_l1_tool`. The shape
  stays. **Only the identity binding changes (see §6).**
- `tests/test_agent_wiring.py` — still validates the graph; expect to add
  one test for the serving-side identity helper.

### 4.2 Replaced

- `agent_server/start_server.py` and the FastAPI/`mlflow.genai.agent_server`
  decorators. Serving endpoints don't run a FastAPI process — the agent is
  the model and MLflow's `ResponsesAgent` serialization handles request
  framing. Replace with `agent_server/responses_agent.py` that subclasses
  `mlflow.pyfunc.ResponsesAgent` and exposes `predict` and `predict_stream`.
- `app.yaml`, the apps frontend, `scripts/start_app.py` — drop for this
  target.
- `agent_server/utils.py:get_user_workspace_client` — the
  `x-forwarded-access-token` header doesn't exist at the serving endpoint.
  Replace with whatever the agreed-on OBO mechanism turns out to be
  (§6) — keep the same function name so the rest of `agent.py` doesn't
  change.

### 4.3 Added

- `agent_server/responses_agent.py` — `ResponsesAgent` subclass. Roughly:

  ```python
  from mlflow.pyfunc import ResponsesAgent
  from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

  class SupervisorResponsesAgent(ResponsesAgent):
      def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
          user_ws = get_user_workspace_client(request)  # §6
          agent = build_l1_agent(user_ws)
          # invoke + collect outputs (sync wrapper around the existing
          # `.ainvoke()` graph)

      def predict_stream(self, request):
          ...
  ```

- `scripts/log_model.py` — logs the model with MLflow, registers it to UC,
  and declares the Databricks resources it needs:

  ```python
  import mlflow
  from mlflow.models.resources import (
      DatabricksGenieSpace, DatabricksServingEndpoint,
  )
  from agent_server.responses_agent import SupervisorResponsesAgent

  with mlflow.start_run():
      mlflow.langchain.log_model(
          lc_model="agent_server.agent",   # or python_model=SupervisorResponsesAgent()
          name="supervisor_example_obo",
          resources=[
              DatabricksGenieSpace(genie_space_id=GENIE_FINANCE_ID),
              DatabricksGenieSpace(genie_space_id=GENIE_SALES_ID),
              DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT),
          ],
          registered_model_name=f"{CATALOG}.{SCHEMA}.supervisor_example_obo",
      )
  ```

  Resource declarations matter: Mosaic AI uses them to provision the
  endpoint's SP with `CAN_RUN` on the Genie spaces and `CAN_QUERY` on the
  LLM endpoint without requiring a human to grant them after deploy.

- `databricks.yml` (new bundle target): declares `registered_models` (UC),
  the bundle variable for `catalog.schema.model_name`, and a
  `serving_endpoints` resource bound to the latest model version with
  `workload_size: Small`, `scale_to_zero_enabled: true`.

- `scripts/deploy_serving.py` — sister to `scripts/deploy.py`. Loads
  `.env`, runs `uv run python -m scripts.log_model` to log and register
  the model, then `databricks bundle deploy -t serving` to update the
  endpoint.

### 4.4 Bundle layout sketch

```yaml
variables:
  uc_catalog:
    default: "main"
  uc_schema:
    default: "default"
  registered_model_name:
    default: "supervisor_example_obo"

targets:
  serving:
    mode: development
    resources:
      registered_models:
        supervisor_model:
          catalog_name: ${var.uc_catalog}
          schema_name: ${var.uc_schema}
          name: ${var.registered_model_name}
      serving_endpoints:
        supervisor_endpoint:
          name: supervisor-example-obo
          config:
            served_entities:
              - entity_name: ${var.uc_catalog}.${var.uc_schema}.${var.registered_model_name}
                entity_version: "${latest}"       # or pinned per-deploy
                workload_size: "Small"
                scale_to_zero_enabled: true
```

## 5. Local dev loop

1. `uv sync`
2. `uv run setup-demo --profile mine` — same as today; provisions the
   two Genie spaces and an MLflow experiment.
3. `uv run log-model --profile mine` — logs the agent and prints the new
   UC model version. (Use the MLflow `predict_stream` locally to smoke
   the graph without standing up an endpoint.)
4. `uv run deploy-serving --profile mine` — bundle deploy of the serving
   target. First deploy takes minutes (endpoint provisioning); updates
   are seconds (new model version, hot-swap).
5. Smoke:

   ```bash
   databricks serving-endpoints query supervisor-example-obo \
     --request '{"input":[{"role":"user","content":"YTD revenue by year?"}]}'
   ```

## 6. OBO at the serving endpoint — *spike results*

**Spike code**: [`spikes/serving-obo/`](../spikes/serving-obo/) — a
one-screen `ResponsesAgent` that probes every plausible identity source
inside `predict()` and runs a live Genie call under both the default
client and the OBO client.

### 6.1 What the docs say

[Databricks Mosaic AI docs](https://docs.databricks.com/aws/en/generative-ai/agent-framework/agent-authentication-model-serving)
define the supported pattern:

```python
# Inside predict() — NOT __init__.
from databricks.sdk import WorkspaceClient
from databricks_ai_bridge import ModelServingUserCredentials

user_client = WorkspaceClient(credentials_strategy=ModelServingUserCredentials())
```

Required at log time:

```python
from mlflow.models.auth_policy import AuthPolicy, UserAuthPolicy

mlflow.pyfunc.log_model(
    ...,
    auth_policy=AuthPolicy(
        user_auth_policy=UserAuthPolicy(
            api_scopes=["dashboards.genie", "sql"],   # OBO scopes
        ),
        system_auth_policy=SystemAuthPolicy(
            resources=[DatabricksGenieSpace(genie_space_id=...)],   # SP grants
        ),
    ),
)
```

OBO-supported resources include **Genie Space** (alongside Vector Search,
Model Serving Endpoint, SQL Warehouse, UC Connections / Tables / Functions,
MCP). For broader OBO needs the docs explicitly say: "Databricks recommends
deploying your agent on Databricks Apps" — i.e. the existing Apps build is
the official recommendation when the resource set is large.

### 6.2 What actually happens on this workspace

Deployed two versions of the `identity_echo` model to a serving endpoint
called `supervisor-obo-spike` on `fevm-stable-po64og`:

**v2** — `mlflow.pyfunc.log_model(..., no auth_policy)`. Query with my
U2M OAuth bearer:

```json
"workspace_client_default": {
  "auth_type": "model-serving",
  "user_name":     "eb8bffce-902c-4a43-aeec-dbbbbaeddf7c",
  "display_name":  "System Service Principal"
},
"context": null,
"env":  {"DATABRICKS_USER_TOKEN": null, ...}    // nothing identity-shaped
```

Endpoint's own SP, no caller identity reachable anywhere.

**v3** — same model, **with** `auth_policy=AuthPolicy(UserAuthPolicy(
api_scopes=["dashboards.genie","sql"]), SystemAuthPolicy([DatabricksGenieSpace(...)]))`.
Query with the same U2M OAuth bearer:

```text
ValueError: model_serving_user_credentials auth: Unable to detect
credentials for user authorization. This error has two common causes:
  (1) Improper OBO configuration — ensure you logged your model with a
      UserAuthPolicy AND that the 'Agent Framework: On-Behalf-Of-User
      Authorization' preview is enabled in your workspace.
  (2) WorkspaceClient instantiation outside of predict()/predict_stream() …
```

We did log with `UserAuthPolicy` and we did instantiate inside `predict()`.
The remaining cause is **(1) — the workspace-level preview flag**:

> "User authorization is in Public Preview. Your workspace admin must
> enable it before you can use user authorization."
> — [docs](https://docs.databricks.com/aws/en/generative-ai/agent-framework/authenticate-on-behalf-of-user)

The preview toggle isn't exposed via the `databricks settings` CLI nor
via any `/api/2.0/previews*` endpoint we could find — it's a UI-only
flip in **Workspace Admin → Settings → Previews → "Agent Framework:
On-Behalf-Of-User Authorization"**.

### 6.3 Conclusion

| Mechanism | Result |
|---|---|
| Default `WorkspaceClient()` inside `predict()` | Endpoint's SP only. No OBO. |
| `request.context` / forwarded env vars | None of them carry the caller. |
| `WorkspaceClient(credentials_strategy=ModelServingUserCredentials())` with `AuthPolicy` attached + preview **disabled** | Raises `ValueError` at runtime. |
| Same, with `AuthPolicy` attached + preview **enabled** (v4 of the spike, deployed *after* the workspace toggle) | ✅ **Works.** See report below. |

v4 identity report (with my U2M OAuth bearer):

```json
"workspace_client_default": {
  "auth_type": "model-serving",
  "user_name":    "981f1bd8-06f8-4d82-95b1-b23fd005b3f7",
  "display_name": "System Service Principal"
},
"workspace_client_obo": {
  "auth_type": "model_serving_user_credentials",
  "user_name":    "lucas.bruand@databricks.com",
  "display_name": "Lucas Bruand"
},
"genie_default": { "status": "error",
                   "message": "failed to reach COMPLETED, got MessageStatus.FAILED" },
"genie_obo":     { "status": "ok",
                   "message_status": "MessageStatus.COMPLETED",
                   "has_attachments": true }
```

The contrast is the proof: the **default SP-bound client fails** the
Genie call (the SP has no Unity Catalog grants on the underlying
tables), while the **OBO client succeeds** under my identity. Per-caller
UC enforcement is intact end-to-end through the serving endpoint.

**Important behavioural note**: the preview is checked at **deploy
time**, not at request time. Re-querying v3 of the spike (logged *before*
the preview was enabled) kept failing with the same error even after the
toggle was on. v4 — exact same model code, logged *after* enabling —
worked on the first request. So enabling the preview requires a
redeploy to take effect.

**Status of the port**: ✅ **feasible on this workspace.** The Apps build
remains the default for production; the serving build becomes a real
option when callers are services / agents instead of humans.

**Risk for customers**: enabling the preview is a per-workspace action,
and it only kicks in for endpoints deployed *after* the toggle. Worth
calling out in the README when this port lands as an explicit
prerequisite, with a link to the Workspace Admin → Settings → Previews
location.

## 7. Resources mapping

The Apps build declares everything in `databricks.yml`. The serving
build splits the declarations: most resources go into the **MLflow
`AuthPolicy` attached at log time** (the spike showed this is the actual
plumbing path), and only the endpoint shape lives in `databricks.yml`.

| Concern | Apps build today | Serving build (confirmed by spike unless marked *guess*) |
|---|---|---|
| Per-caller token | Apps proxy injects `x-forwarded-access-token`; agent reads via `get_request_headers()` | `WorkspaceClient(credentials_strategy=ModelServingUserCredentials())` inside `predict()` — instantiation **must** be inside `predict()` / `predict_stream()`. |
| OBO scopes | `user_api_scopes: [dashboards.genie, sql]` in `databricks.yml` | `UserAuthPolicy(api_scopes=["dashboards.genie", "sql"])` at log time, inside `mlflow.pyfunc.log_model(..., auth_policy=AuthPolicy(user_auth_policy=...))`. |
| Genie space access (end-user) | `genie_space:` resource (CAN_RUN for the app's SP) in `databricks.yml` | Covered by `UserAuthPolicy` scopes above. The user must already have CAN_RUN; no extra grant from the model. |
| Genie space access (endpoint SP) | n/a — Apps use forwarded token only | `SystemAuthPolicy(resources=[DatabricksGenieSpace(genie_space_id=...)])` at log time. Used for any code path that uses the default `WorkspaceClient()`. |
| LLM endpoint access | App `serving_endpoint:` resource (CAN_QUERY) in `databricks.yml` | `SystemAuthPolicy(resources=[DatabricksServingEndpoint(endpoint_name=...)])` at log time. |
| Experiment | App `experiment:` resource | *Guess:* same — declare on the bundle (not the endpoint). MLflow tracking just needs the URI; not bundled into AuthPolicy. |
| Endpoint shape (workload size, scale-to-zero, traffic) | n/a (Apps shape lives in `app.yaml` + DAB) | `serving_endpoints:` DAB resource in `databricks.yml` — sketch in §4.4, **untested**. |

## 8. What we lose

- The bundled chat UI. Customers using a browser would talk to the
  endpoint via **AI Playground** or the **MLflow Review App**.
- Per-request browser-OAuth context (Apps' `x-forwarded-access-token` is
  the cleanest OBO surface Databricks ships). Whatever mechanism §6
  picks will be more constrained.
- Mid-request streaming over WebSocket — serving endpoints support SSE
  via `predict_stream`, so streaming itself stays.

## 9. Acceptance criteria

- [ ] §6 spike: caller identity is reachable inside the served model.
      Document which mechanism worked and link to the Databricks doc.
- [ ] `uv run deploy-serving --profile mine` creates / updates the
      endpoint end-to-end on a clean checkout (idempotent).
- [ ] `databricks serving-endpoints query supervisor-example-obo …`
      returns a routed answer for a finance prompt **and** for a sales
      prompt.
- [ ] Two end-users with different UC grants on the same Genie space see
      different result sets via the endpoint (proves OBO survived).
- [ ] Adding a third domain still only requires editing `DOMAINS` in
      `agent.py`, adding the prompt, and adding the resource to the
      `log_model` call. No graph rewiring.
- [ ] `evaluate_agent.py` runs against the endpoint URL the same way it
      runs against `localhost:8000` today (swap `predict_fn`).

## 10. Open questions

- ~~Which OBO mechanism (§6) actually works on the target workspace?~~
  **Resolved.** `ModelServingUserCredentials` + `AuthPolicy` works once
  the workspace OBO preview is on, *and* the model is redeployed after
  the toggle. v4 of the spike confirmed `workspace_client_obo` resolves
  to the caller and a Genie call under it succeeds while the SP path
  fails on the same call.
- Where does the agent run when the endpoint scales to zero — cold start
  cost relative to the warmed Apps process. Measure before promising a
  customer "this is faster."
- `mlflow.langchain.log_model` vs `mlflow.pyfunc.log_model` with a
  hand-written `ResponsesAgent` subclass: the first is shorter; the
  second is more explicit. Default to subclassing if the LangChain
  autolog path drops trace metadata.
- Are Genie spaces and the LLM endpoint the only resources the agent
  touches, or should we also declare `DatabricksTable` for the
  underlying UC tables (Mosaic AI permissions docs are inconsistent)?
- Can a single bundle host **both** the App target and the Serving
  target, with shared `variables:` (Genie space IDs, experiment id)? If
  yes, that's what to ship; if no, fork the bundle.
