# Hierarchical Supervisor Agent on Databricks Apps (with Genie OBO)

## 1. Goal

Build a Databricks App that exposes a custom hierarchical LangChain/LangGraph
agent. The agent has two layers of supervisors, and the leaves are Genie
spaces that must be queried **on-behalf-of (OBO)** the calling end-user — i.e.
data access enforced with the user's Unity Catalog permissions, not the app's
service-principal.

```
                ┌────────────────────────────────────┐
                │       Supervisor L1 (router)       │
                │  routes by domain / business unit  │
                └───────────────┬────────────────────┘
                                │ handoff
                ┌───────────────┴─────────────────┐
                │                                 │
        ┌───────▼───────┐                ┌────────▼──────┐
        │ Supervisor L2 │                │ Supervisor L2 │
        │   (Finance)   │      ...       │   (Sales)     │
        └───────┬───────┘                └───────┬───────┘
                │ tool call (OBO)                │ tool call (OBO)
        ┌───────▼───────┐                ┌───────▼───────┐
        │ Genie space A │                │ Genie space B │
        │ (finance KPI) │                │ (sales pipe)  │
        └───────────────┘                └───────────────┘
```

End-user → App (UI / `/responses` API) → L1 supervisor → L2 supervisor →
Genie tool (uses **user token**, not SP token).

## 2. Tech stack

| Concern              | Choice                                                                  |
| -------------------- | ----------------------------------------------------------------------- |
| Language             | Python ≥ 3.11                                                           |
| Package manager      | `uv`                                                                    |
| Agent framework      | `langchain` + `langgraph` (`langchain.agents.create_agent`)             |
| LLM                  | `databricks_langchain.ChatDatabricks` (e.g. `databricks-gpt-5-2`)       |
| Server               | MLflow `ResponsesAgent` via `mlflow.genai.agent_server` (FastAPI)       |
| Tracing              | MLflow autolog → MLflow experiment                                      |
| Genie integration    | Databricks MCP server: `/api/2.0/mcp/genie/{space_id}`                  |
| Identity (leaf calls)| OBO — `x-forwarded-access-token` → user `WorkspaceClient`               |
| Identity (LLM, infra)| App service-principal                                                   |
| Packaging / deploy   | Databricks Asset Bundles (`databricks.yml` + `app.yaml` + `manifest.yaml`) |
| Base template        | [`databricks/app-templates/agent-langgraph`](https://github.com/databricks/app-templates/tree/main/agent-langgraph) |

## 3. Repository layout

Mirrors `agent-langgraph` so the bundled chat UI, `start-app`, and quickstart
scripts work out of the box.

```
supervisor-example-obo/
├── SPECS/
│   └── SPEC.md                 # this document
├── agent_server/
│   ├── __init__.py
│   ├── agent.py                # hierarchical supervisor graph
│   ├── prompts.py              # L1 / L2 system prompts
│   ├── utils.py                # OBO helpers, stream adapters
│   ├── start_server.py         # FastAPI bootstrap
│   └── evaluate_agent.py       # offline eval (optional)
├── scripts/
│   ├── quickstart.py
│   └── start_app.py
├── app.yaml                    # Databricks App runtime config
├── databricks.yml              # DAB definition
├── manifest.yaml               # App manifest (resources + user_api_scopes)
├── pyproject.toml              # uv / hatchling project
├── .env.example
└── README.md
```

## 4. Agent design

### 4.1 LangGraph hierarchical supervisor

Use the LangGraph **supervisor pattern**: each supervisor is an agent whose
"tools" are calls into child agents (handoffs). Two layers:

- **L1 router** — small prompt, picks one of N domain supervisors.
- **L2 domain supervisor** — domain-aware system prompt, has access to one or
  more Genie spaces (as tools) plus optional UC-function tools.

Each supervisor is constructed with `langchain.agents.create_agent(tools, model)`
and exposed to its parent as a `@tool` that takes a free-form question and
returns the child's final answer (string + structured artifacts).

### 4.2 Genie tool (leaf)

The Genie leaf calls the **direct Genie REST API** via the SDK, with the
client bound to the end-user's identity:

```python
@tool(name_or_callable=f"genie_{name}")
def _query(question: str) -> str:
    msg = user_ws.genie.start_conversation_and_wait(space_id, question)
    # extract text + SQL from msg.attachments, return as one string
```

Each L2 supervisor gets exactly one tool, `genie_<domain>`. Because the
`WorkspaceClient` was constructed from the forwarded user token, the Genie
conversation runs under that user's UC grants.

**Why not the MCP route** (`/api/2.0/mcp/genie/{space_id}`): we tried it
first. The Databricks Apps `user_api_scopes` allowlist is limited
(documented values: `sql`, `dashboards.genie`, `files.files`), and the MCP
endpoint requires a broader scope (`all-apis`, which the bundle validator
refuses). So the forwarded user token always 403'd against MCP. The direct
Genie API works fine with just `dashboards.genie`. See §12.

### 4.3 Request lifecycle

1. Client POSTs `/responses` (or `/invocations`) with
   `input=[{"role":"user","content":...}]`.
2. `stream_handler` reads request headers via `get_request_headers()`.
3. Build a **user `WorkspaceClient`** from `x-forwarded-access-token`.
4. Instantiate the graph: L1 supervisor with L2 supervisors as tools; each L2
   supervisor receives a Genie tool bound to the user's `WorkspaceClient`.
5. Stream `agent.astream(..., stream_mode=["updates","messages"])` and emit
   `ResponsesAgentStreamEvent`s through `process_agent_astream_events`.

## 5. OBO authentication

Follow the pattern from `agent-langgraph/agent_server/utils.py`:

```python
from mlflow.genai.agent_server import get_request_headers
from databricks.sdk import WorkspaceClient

def get_user_workspace_client() -> WorkspaceClient:
    token = get_request_headers().get("x-forwarded-access-token")
    return WorkspaceClient(token=token, auth_type="pat")
```

Rules:

- LLM calls (`ChatDatabricks`) — use the SP `WorkspaceClient` (default).
- Genie / data-plane calls — **must** use the user `WorkspaceClient`.
- Local dev — when running outside Apps, `x-forwarded-access-token` is absent;
  fall back to the developer's CLI profile (`WorkspaceClient()`), guarded by
  an env flag (`OBO_FALLBACK_TO_DEFAULT=1`) so prod can't silently degrade.
- `databricks.yml` must declare `user_api_scopes` at the app resource
  level so the app is allowed to request a downscoped user token (see §6).
  `manifest.yaml`'s `user_api_scopes` block is not enough on its own —
  DABs reads from `databricks.yml` and what gets to the deployed app's
  `effective_user_api_scopes` comes from there.

## 6. Databricks resources

### 6.1 `manifest.yaml`

```yaml
version: 1
name: "Supervisor Agent (OBO)"
description: "Hierarchical LangGraph supervisor agent querying Genie spaces on behalf of the calling user."

resource_specs:
  - name: "experiment"
    description: "MLflow experiment for agent traces."
    experiment_spec:
      permission: "CAN_EDIT"

  - name: "genie_finance"
    description: "Finance Genie space."
    genie_space_spec:
      permission: "CAN_RUN"

  - name: "genie_sales"
    description: "Sales Genie space."
    genie_space_spec:
      permission: "CAN_RUN"

  - name: "llm_endpoint"
    description: "Chat model serving endpoint used by all supervisors."
    serving_endpoint_spec:
      permission: "CAN_QUERY"

user_api_scopes:
  - "dashboards.genie"   # required for Genie OBO calls
  - "sql"                # Genie issues SQL warehouse queries under the hood
```

> The `user_api_scopes` allowlist is narrower than general Databricks OAuth
> scopes. Documented values: `sql`, `dashboards.genie`, `files.files`.
> `all-apis` is **not** accepted here — the bundle validator rejects it.

### 6.2 `app.yaml`

```yaml
command: ["uv", "run", "start-app"]

env:
  - name: MLFLOW_TRACKING_URI
    value: "databricks"
  - name: MLFLOW_REGISTRY_URI
    value: "databricks-uc"
  - name: API_PROXY
    value: "http://localhost:8000/invocations"
  - name: MLFLOW_EXPERIMENT_ID
    valueFrom: "experiment"
  - name: GENIE_FINANCE_SPACE_ID
    valueFrom: "genie_finance"
  - name: GENIE_SALES_SPACE_ID
    valueFrom: "genie_sales"
  - name: LLM_ENDPOINT
    valueFrom: "llm_endpoint"
```

### 6.3 `databricks.yml` (DABs)

```yaml
bundle:
  name: supervisor_example_obo

resources:
  apps:
    supervisor_example_obo:
      name: "supervisor-example-obo"
      description: "Hierarchical supervisor agent (LangGraph) with Genie OBO"
      source_code_path: ./
      config:
        command: ["uv", "run", "start-app"]
        env:
          - { name: MLFLOW_TRACKING_URI,  value: "databricks" }
          - { name: MLFLOW_REGISTRY_URI,  value: "databricks-uc" }
          - { name: API_PROXY,            value: "http://localhost:8000/invocations" }
          - { name: CHAT_APP_PORT,        value: "3000" }
          - { name: CHAT_PROXY_TIMEOUT_SECONDS, value: "300" }
          - { name: MLFLOW_EXPERIMENT_ID, value_from: "experiment" }
          - { name: GENIE_FINANCE_SPACE_ID, value_from: "genie_finance" }
          - { name: GENIE_SALES_SPACE_ID,   value_from: "genie_sales" }
          - { name: LLM_ENDPOINT,           value_from: "llm_endpoint" }

      resources:
        - name: "experiment"
          experiment:
            experiment_id: ""           # filled by quickstart
            permission: "CAN_MANAGE"
        - name: "genie_finance"
          genie_space:
            name: "Finance Genie"
            space_id: "<TODO-FINANCE-SPACE-ID>"
            permission: "CAN_RUN"
        - name: "genie_sales"
          genie_space:
            name: "Sales Genie"
            space_id: "<TODO-SALES-SPACE-ID>"
            permission: "CAN_RUN"
        - name: "llm_endpoint"
          serving_endpoint:
            name: "databricks-gpt-5-2"
            permission: "CAN_QUERY"

targets:
  dev:
    mode: development
    default: true
  prod:
    mode: production
    resources:
      apps:
        supervisor_example_obo:
          name: supervisor-example-obo
```

## 7. `pyproject.toml`

Minimum dependencies (copy from `agent-langgraph`, drop nothing; this list is
the additive baseline):

```toml
[project]
name = "supervisor-example-obo"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.129.0",
  "uvicorn>=0.41.0",
  "databricks-sdk>=0.40.0",
  "databricks-langchain>=0.17.0",
  "databricks-agents>=1.9.3",
  "mlflow>=3.10.0",
  "langchain>=1.0.0",
  "langgraph>=1.1.0",
  "langchain-mcp-adapters>=0.2.1",
  "python-dotenv>=1.2.1",
]

[project.scripts]
start-app   = "scripts.start_app:main"
start-server = "agent_server.start_server:main"
quickstart  = "scripts.quickstart:main"
```

## 8. Key implementation sketches

### 8.1 `agent_server/agent.py` (current shape)

```python
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent
from langchain_core.tools import tool

from agent_server import prompts
from agent_server.utils import get_user_workspace_client

LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "databricks-gpt-5-2")

DOMAINS = [
    {"name": "finance", "space_id_env": "GENIE_FINANCE_SPACE_ID",
     "system_prompt": prompts.FINANCE_L2, "tool_description": "..."},
    {"name": "sales",   "space_id_env": "GENIE_SALES_SPACE_ID",
     "system_prompt": prompts.SALES_L2,   "tool_description": "..."},
]

def _build_genie_tool(user_ws, name, space_id):
    @tool(name_or_callable=f"genie_{name}")
    def _query(question: str) -> str:
        msg = user_ws.genie.start_conversation_and_wait(space_id, question)
        # extract text/SQL from msg.attachments, return one string
        ...
    return _query

def _build_l2_supervisor(user_ws, domain):
    space_id = os.environ[domain["space_id_env"]]
    return create_agent(
        tools=[_build_genie_tool(user_ws, domain["name"], space_id)],
        model=ChatDatabricks(endpoint=LLM_ENDPOINT),
        system_prompt=domain["system_prompt"],
    )

def build_l1_agent(user_ws):
    handoff_tools = []
    for d in DOMAINS:
        l2 = _build_l2_supervisor(user_ws, d)

        @tool(name_or_callable=f"ask_{d['name']}", description=d["tool_description"])
        async def _ask(question: str, _l2=l2) -> str:
            r = await _l2.ainvoke({"messages": [("user", question)]})
            return r["messages"][-1].content

        handoff_tools.append(_ask)

    return create_agent(tools=handoff_tools,
                        model=ChatDatabricks(endpoint=LLM_ENDPOINT),
                        system_prompt=prompts.L1_ROUTER)

@stream()
async def stream_handler(request):
    user_ws = get_user_workspace_client()
    agent = build_l1_agent(user_ws)          # sync; no MCP to await
    async for ev in process_agent_astream_events(
        agent.astream({"messages": to_chat_completions_input(...)},
                      stream_mode=["updates", "messages"])
    ):
        yield ev
```

Notes:
- `create_agent` in `langchain` 1.x takes `system_prompt=`, not `prompt=`
  (the first deploy 500'd until we fixed this).
- L1 / L2 builders are **sync** — no async MCP setup to await.
- L2 supervisors get a single `genie_<domain>` tool. L1 sees one
  `ask_<domain>` tool per domain.

### 8.2 Prompts (`agent_server/prompts.py`)

- `L1_ROUTER` — "You are a router. Choose exactly one specialist via the
  provided tools. Do not answer directly."
- `FINANCE_L2`, `SALES_L2` — domain context, instruction to call the Genie
  tool, and guardrails on what they may not infer outside their domain.

## 9. Local dev loop

1. `uv sync`
2. `databricks auth login` → set `DATABRICKS_CONFIG_PROFILE` in `.env`
3. `cp .env.example .env` and fill `MLFLOW_EXPERIMENT_ID`, `GENIE_*_SPACE_ID`,
   `LLM_ENDPOINT`, `OBO_FALLBACK_TO_DEFAULT=1`.
4. `uv run start-app` — agent at `http://localhost:8000`, chat UI at `:3000`.
5. Test:
   ```bash
   curl -X POST localhost:8000/responses -H 'Content-Type: application/json' \
     -d '{"input":[{"role":"user","content":"YTD revenue by region"}]}'
   ```

## 10. Deployment (DABs)

```bash
databricks bundle validate
databricks bundle deploy -t dev
databricks bundle run supervisor_example_obo -t dev   # starts the app
```

After deploy, requests to `https://<app>.databricksapps.com/responses` carry
`x-forwarded-access-token` automatically when the caller authenticates via
the Apps OAuth flow; Genie calls then run as that user.

### 10.1 Reference deployment (current)

- Workspace: `https://fevm-serverless-stable-po64og.cloud.databricks.com`
- App URL: https://supervisor-example-obo-7474659269459324.aws.databricksapps.com
- Bundle target: `dev`
- MLflow experiment: `1456842180535736`
- Genie spaces:
  - finance `01f15474886a16d5aa027e0791fa855a` — `samples.tpch.{orders, lineitem, customer, nation, region}`
  - sales   `01f15474da541750b15a234e1fcc4145` — `samples.bakehouse.sales_{transactions, customers, franchises, suppliers}`
- SP client_id: `2a384f0d-ceaf-4a6d-878a-77d42dfbd954`
- `effective_user_api_scopes`: `['sql', 'iam.current-user:read', 'dashboards.genie', 'iam.access-control:read']`

### 10.2 Logs require OAuth

`databricks apps logs <app>` only accepts an OAuth profile, not a PAT.
Set up a secondary U2M profile (`databricks auth login -p <name> --host …`)
to read runtime logs.

### 10.3 Genie spaces were created via `/api/2.0/data-rooms`

The public SDK has no `create_space`; we POSTed to `/api/2.0/data-rooms`
with `display_name`, `warehouse_id`, and `table_identifiers`. The newer
`/api/2.0/genie/spaces` endpoint requires a `serialized_space` proto and
isn't usable for fresh creation. Worth noting in case the older endpoint
gets deprecated.

## 11. Acceptance criteria

- [x] `databricks bundle validate` passes on a fresh checkout.
- [x] `uv run start-server` boots locally and `/responses` returns a
      non-empty answer routed through L1 → L2 → Genie. Verified against
      `fevm-stable-po64og`:
      - finance: `samples.tpch.orders` total revenue =
        **1,133,439,215,246.25**
      - sales: `samples.bakehouse.sales_transactions` total count = **3,333**
- [x] Adding a third domain requires only: a new Genie space resource in
      `databricks.yml`/`manifest.yaml`, a new prompt, and one extra entry
      in `DOMAINS` in `agent.py` — no graph rewiring.
- [ ] Two end-users with **different** UC grants on the same Genie space
      see different result sets for the same question (proves OBO).
      *Not exercised yet — requires a second test user with restricted
      grants on the bakehouse / tpch tables.*
- [ ] MLflow traces show the three nested spans: L1 supervisor, L2
      supervisor, Genie tool call. *Traces are reaching experiment
      `1456842180535736`; visual inspection of the span hierarchy pending.*

## 12. Decisions

- **Token scope** *(resolved)* — final `user_api_scopes` is
  `[dashboards.genie, sql]`. The Apps allowlist is narrow (documented:
  `sql / dashboards.genie / files.files`); `all-apis` is rejected by the
  bundle validator. The actually-deployed user token gets a few IAM scopes
  added on top (see §10.1).
- **MCP vs. direct Genie SDK** *(switched mid-flight)* — original plan was
  MCP first. Reality: the MCP route
  (`/api/2.0/mcp/genie/{space_id}`) 403s under OBO because it needs a
  scope outside the `user_api_scopes` allowlist. Confirmed by decoding the
  working U2M CLI token — it carries `all-apis`, which `user_api_scopes`
  refuses. Switched the leaf to `w.genie.start_conversation_and_wait`,
  which works with just `dashboards.genie`. Reconsider MCP only when the
  Apps allowlist grows.
- **Conversation memory** — out of scope for v1. Each request rebuilds the
  graph and is stateless. If memory is later required, port the
  short-term-memory checkpointer pieces from `agent-langgraph-advanced`
  rather than rolling a new design.
- **Eval harness** — `evaluate_agent.py` must include at least one OBO
  negative case: a test user with no UC grant on the underlying tables. The
  agent must surface the permission error from Genie back to the caller and
  must **not** retry under SP credentials. The `OBO_FALLBACK_TO_DEFAULT`
  flag is local-dev only and must be off in eval/prod.
