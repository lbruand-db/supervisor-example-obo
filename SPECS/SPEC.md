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

The Genie tool wraps the Databricks **MCP Genie server**:

```python
DatabricksMCPServer(
    name=f"genie-{space_name}",
    url=f"{host}/api/2.0/mcp/genie/{space_id}",
    workspace_client=user_ws,   # <-- OBO client, see §5
)
```

`DatabricksMultiServerMCPClient.get_tools()` returns LangChain tools the L2
supervisor can call. Tool calls hit Genie under the user's identity, so the
underlying SQL warehouse query is governed by Unity Catalog grants on the
calling user.

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
- `manifest.yaml` must declare `user_api_scopes` so the app is allowed to
  request a downscoped user token (see §6).

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

### 8.1 `agent_server/agent.py` (pseudocode)

```python
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks, DatabricksMCPServer, DatabricksMultiServerMCPClient
from langchain.agents import create_agent
from langchain_core.tools import tool

from agent_server.utils import get_user_workspace_client, host_from_env

LLM = ChatDatabricks(endpoint=os.environ["LLM_ENDPOINT"])

def _genie_tools(user_ws: WorkspaceClient, space_id: str, name: str):
    client = DatabricksMultiServerMCPClient([
        DatabricksMCPServer(
            name=f"genie-{name}",
            url=f"{host_from_env()}/api/2.0/mcp/genie/{space_id}",
            workspace_client=user_ws,
        )
    ])
    return await client.get_tools()

async def build_l2(user_ws, space_id, name, prompt):
    tools = await _genie_tools(user_ws, space_id, name)
    return create_agent(tools=tools, model=LLM, prompt=prompt)

async def build_l1(user_ws):
    finance = await build_l2(user_ws, os.environ["GENIE_FINANCE_SPACE_ID"],
                             "finance", prompts.FINANCE_L2)
    sales   = await build_l2(user_ws, os.environ["GENIE_SALES_SPACE_ID"],
                             "sales",   prompts.SALES_L2)

    @tool
    async def ask_finance(question: str) -> str:
        """Use for finance / accounting / revenue questions."""
        result = await finance.ainvoke({"messages": [("user", question)]})
        return result["messages"][-1].content

    @tool
    async def ask_sales(question: str) -> str:
        """Use for pipeline, opportunities, and sales-ops questions."""
        result = await sales.ainvoke({"messages": [("user", question)]})
        return result["messages"][-1].content

    return create_agent(tools=[ask_finance, ask_sales], model=LLM,
                        prompt=prompts.L1_ROUTER)

@stream()
async def stream_handler(request):
    user_ws = get_user_workspace_client()
    agent = await build_l1(user_ws)
    async for ev in process_agent_astream_events(
        agent.astream({"messages": to_chat_completions_input(...)},
                      stream_mode=["updates", "messages"])
    ):
        yield ev
```

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

## 11. Acceptance criteria

- [ ] `databricks bundle validate` passes on a fresh checkout.
- [ ] `uv run start-app` boots locally and `/responses` returns a non-empty
      answer routed through L1 → L2 → Genie.
- [ ] Two end-users with **different** UC grants on the same Genie space see
      different result sets for the same question (proves OBO).
- [ ] MLflow traces show the three nested spans: L1 supervisor, L2
      supervisor, Genie tool call.
- [ ] Adding a third domain requires only: a new Genie space resource in
      `databricks.yml`/`manifest.yaml`, a new prompt, and one extra `@tool`
      wrapper in `agent.py` — no graph rewiring.

## 12. Decisions

- **Token scope** — start with `dashboards.genie` + `sql` in
  `user_api_scopes`; the exact Genie scope name will be confirmed on first
  deploy (`databricks bundle run -t dev` and inspect the granted scopes). If
  the platform reports an unknown-scope error, adjust to whatever name the
  deploy validation surfaces.
- **MCP vs. direct Genie SDK** — implement against the Databricks MCP Genie
  server (`/api/2.0/mcp/genie/{space_id}`). Only fall back to
  `WorkspaceClient.genie.start_conversation` if MCP latency proves
  unacceptable in eval; do not build both paths up front.
- **Conversation memory** — out of scope for v1. Each request rebuilds the
  graph and is stateless. If memory is later required, port the
  short-term-memory checkpointer pieces from `agent-langgraph-advanced`
  rather than rolling a new design.
- **Eval harness** — `evaluate_agent.py` must include at least one OBO
  negative case: a test user with no UC grant on the underlying tables. The
  agent must surface the permission error from Genie back to the caller and
  must **not** retry under SP credentials. The `OBO_FALLBACK_TO_DEFAULT`
  flag is local-dev only and must be off in eval/prod.
