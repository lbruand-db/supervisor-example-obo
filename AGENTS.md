# Agent Development Guide

## This project at a glance

Hierarchical supervisor agent. Two deployment surfaces share the **same
agent code** in `agent_server/`:

| Surface | Entry point | OBO mechanism | When to pick |
|---|---|---|---|
| **Databricks Apps** (default) | `uv run deploy` → app at `…databricksapps.com` | `x-forwarded-access-token` lifted by `utils.get_user_workspace_client` | Browser users; bundled chat UI; no preview prereq. |
| **Model Serving** | `uv run deploy-serving` → endpoint at `…/serving-endpoints/...` | `WorkspaceClient(credentials_strategy=ModelServingUserCredentials())` inside `responses_agent.py:predict()` | Service-to-service callers; UC-governed model registration; scale-to-zero. **Requires** the workspace public preview *"Agent Framework: On-Behalf-Of-User Authorization"* and a (re)deploy after enabling it. |

Both surfaces:

- **L1 router** picks one domain. **L2 domain supervisors** (finance, sales)
  each own one Genie space.
- **OBO at the leaf**: L2 supervisors call Genie under the **end-user's
  identity**. LLM + infra calls stay on the SP.
- **Genie via the direct SDK**, not MCP. The MCP route
  (`/api/2.0/mcp/genie/{space_id}`) 403s under OBO because it requires a
  scope outside the `user_api_scopes` allowlist (`sql`, `dashboards.genie`,
  `files.files`). The leaf uses `w.genie.start_conversation_and_wait(...)`,
  which works with just `dashboards.genie`.

### Live deployment

- Workspace: `https://fevm-serverless-stable-po64og.cloud.databricks.com`
- CLI profile (PAT): `fevm-stable-po64og`. OAuth profile (needed for
  `databricks apps logs`): `fevm-po64og-oauth`.
- App URL: https://supervisor-example-obo-7474659269459324.aws.databricksapps.com
- Bundle name: `supervisor_example_obo` (DAB resource id), app name:
  `supervisor-example-obo`.
- Genie spaces (created via `/api/2.0/data-rooms`, since the SDK has no
  `create_space`): finance `01f15474886a16d5aa027e0791fa855a`
  (samples.tpch.*), sales `01f15474da541750b15a234e1fcc4145`
  (samples.bakehouse.*).
- MLflow experiment: `1456842180535736`.

### Spec / source-of-truth

- Design + decisions live in [`SPECS/SPEC.md`](SPECS/SPEC.md). Update it
  when the architecture or scope decisions change.
- Tests: `uv run pytest` (smoke tests only — no LLM/Genie calls).
- Lint/format: `uv run ruff check .` / `uv run ruff format .`.
- CI: `.github/workflows/ci.yml` runs ruff + pytest on every PR/push.
- Deploy: `.github/workflows/deploy.yml` (manual dispatch, OIDC).

### Editing rules of thumb

- **Don't reintroduce the MCP Genie path** unless `user_api_scopes` grows
  to include the right scope. The current SDK leaf is the workaround.
- **Don't widen `user_api_scopes`** beyond `dashboards.genie + sql` without
  a real need — the OBO surface should stay minimal.
- **Adding a domain**: append an entry to `DOMAINS` in
  `agent_server/agent.py`, add the system prompt to `prompts.py`, and
  declare the new `genie_space` resource in `databricks.yml` /
  `manifest.yaml`. Nothing else should need to change.
- **Don't silently bypass OBO**: `OBO_FALLBACK_TO_DEFAULT=1` is a local-dev
  escape hatch only — must be unset in eval/prod.

---

## MANDATORY First Actions

**Ask the user interactively:**

1. **App deployment target:**
   > "Do you have an existing Databricks app you want to deploy to, or should we create a new one? If existing, what's the app name?"

   *Note: New apps should use the `agent-*` prefix (e.g., `agent-data-analyst`) unless the user specifies otherwise.*

2. **If the user mentions memory, conversation history, or persistence:**
   > "For memory capabilities, do you have an existing Lakebase instance? If so, what's the instance name?"

**Then set up the environment using quickstart:**

1. **Read the quickstart skill** at `.claude/skills/quickstart/SKILL.md` — it contains all available CLI flags, what the script configures, and fallback instructions.
2. **Check if `.env` exists.** If it does, the environment is already configured — read it to find `DATABRICKS_CONFIG_PROFILE` and skip to verifying auth. If `.env` does not exist, run quickstart:
   ```bash
   uv run quickstart --profile <profile-name>
   ```
3. Run `databricks auth profiles` to verify the profile is configured and valid.

**CRITICAL: All `databricks` CLI commands must include the profile from `.env`.** Either use `--profile` or set the env var:

```bash
databricks <command> --profile <profile>
# or
DATABRICKS_CONFIG_PROFILE=<profile> databricks <command>
```

> **Why this matters:** Without the profile, the CLI may target the wrong workspace, causing "not found" errors for experiments, apps, or other resources.

## Understanding User Goals

**Ask the user questions to understand what they're building:**

1. **What is the agent's purpose?** (e.g., data analyst assistant, customer support, code helper)
2. **What data or tools does it need access to?**
   - Databases/tables (Unity Catalog)
   - Documents for RAG (Vector Search)
   - Natural language data queries (Genie Spaces)
   - External APIs or services
3. **Any specific Databricks resources they want to connect?**

Use `uv run discover-tools` to show them available resources in their workspace, then help them select the right ones for their use case. **See the `add-tools` skill for how to connect tools and grant permissions.**

## Handling Deployment Errors

**If `databricks bundle deploy` fails with "An app with the same name already exists":**

Ask the user: "I see there's an existing app with the same name. Would you like me to bind it to this bundle so we can manage it, or delete it and create a new one?"

- **If they want to bind**: See the **deploy** skill for binding steps
- **If they want to delete**: Run `databricks apps delete <app-name>` then deploy again

## Supervisor API (Offloading the Agent Loop)

The **Supervisor API** lets Databricks run the tool-selection and agent loop server-side. Declare hosted tools (Genie spaces, UC functions, Knowledge Assistants, UC connection MCP servers, Databricks App endpoints) and call `responses.create()` — Databricks handles the rest.

**Use when the user wants to** connect Genie spaces, UC functions, or other Databricks-hosted tools without managing the agent loop themselves.

**Limitations:**
- Tools run as the app's service principal (no user token forwarding) — grant permissions in `databricks.yml`
- Cannot mix hosted tools with client-side function tools in the same request
- Inference parameters (`temperature`, `top_p`, etc.) are not supported when tools are passed
- `stream` and `background` cannot both be `true` in the same request
- Background mode has a maximum execution time of 30 minutes

**Skills:**
- Use **supervisor-api** to set up the Supervisor API with hosted tools
- Use **supervisor-api-background-mode** for tasks that may exceed HTTP timeout limits (complex multi-tool workflows, large data analysis)

## Agent Evaluation

When the user asks about evaluating their agent (quality, metrics, scorers, datasets, or tracing), suggest installing the **MLflow Skills** from https://github.com/mlflow/skills. These provide expert guidance for evaluation workflows using MLflow's native APIs.

**Relevant skills:**
- **agent-evaluation** — end-to-end evaluation: dataset creation, scorer selection, execution, result analysis
- **instrumenting-with-mlflow-tracing** — set up automatic tracing for debugging and observability
- **analyze-mlflow-trace** — examine span data and assessments to identify issues

**Install command:**
```bash
npx skills add mlflow/skills
```

After installation, the skills will be available as slash commands (e.g., `/agent-evaluation`). This template also includes a built-in `evaluate_agent.py` script — run it with `uv run agent-evaluate` after starting the local server.

---

## Available Skills

**Before executing any task, read the relevant skill file in `.claude/skills/`** - they contain tested commands, patterns, and troubleshooting steps.

| Task | Skill | Path |
|------|-------|------|
| Setup, auth, first-time | **quickstart** | `.claude/skills/quickstart/SKILL.md` |
| Find tools/resources | **discover-tools** | `.claude/skills/discover-tools/SKILL.md` |
| Create tool resources | **create-tools** | `.claude/skills/create-tools/SKILL.md` |
| Deploy to Databricks | **deploy** | `.claude/skills/deploy/SKILL.md` |
| Add tools & permissions | **add-tools** | `.claude/skills/add-tools/SKILL.md` |
| Run/test locally | **run-locally** | `.claude/skills/run-locally/SKILL.md` |
| Modify agent code | **modify-agent** | `.claude/skills/modify-agent/SKILL.md` |
| Configure Lakebase storage | **lakebase-setup** | `.claude/skills/lakebase-setup/SKILL.md` |
| Add memory capabilities | **agent-memory** | `.claude/skills/agent-memory/SKILL.md` |
| Offload agent loop to Databricks | **supervisor-api** | `.claude/skills/supervisor-api/SKILL.md` |
| Long-running background tasks | **supervisor-api-background-mode** | `.claude/skills/supervisor-api-background-mode/SKILL.md` |

**Note:** All agent skills are located in `.claude/skills/` directory.

> **Adding Memory?** The **lakebase-setup** and **agent-memory** skills help you add conversation history or persistent user memory to this agent. For pre-configured memory, see the `agent-langgraph-advanced` template.

---

## Quick Commands

| Task | Command |
|------|---------|
| Setup | `uv run quickstart` |
| Discover tools | `uv run discover-tools` |
| Run locally | `uv run start-app` |
| Deploy (Apps) | `uv run deploy --profile <p>` |
| Deploy (Model Serving) | `uv run deploy-serving --profile <p>` |
| View Apps logs | `databricks apps logs <app-name> --follow -p <oauth-profile>` |
| View Serving logs | `databricks serving-endpoints logs <endpoint-name> -p <oauth-profile>` |

---

## Key Files

| File | Purpose |
|------|---------|
| `agent_server/agent.py` | Hierarchical supervisor graph: L1 router, L2 domain supervisors, Genie leaf tools (direct SDK, OBO-bound) |
| `agent_server/prompts.py` | L1 router + L2 domain system prompts |
| `agent_server/utils.py` | `get_user_workspace_client()` (OBO with `OBO_FALLBACK_TO_DEFAULT` escape hatch), stream helpers |
| `agent_server/start_server.py` | FastAPI server + MLflow setup |
| `agent_server/evaluate_agent.py` | MLflow eval scaffolding — finance/sales/OBO-denial cases |
| `databricks.yml` | DAB definition: app name, env, `user_api_scopes`, Genie + experiment + serving-endpoint resources |
| `manifest.yaml` | Standalone app manifest (mirrors `databricks.yml` resources) |
| `tests/test_agent_wiring.py` | Smoke tests for DOMAINS shape, prompt guardrails, OBO contract, L1 tool wiring |
| `.github/workflows/ci.yml` | PR/push CI: ruff + pytest |
| `.github/workflows/deploy.yml` | Manual deploy via OIDC federation |
| `agent_server/responses_agent.py` | MLflow `ResponsesAgent` subclass used by the Model Serving build — same graph, identity bound via `ModelServingUserCredentials` |
| `scripts/setup_demo.py` | `uv run setup-demo`: creates two Genie spaces + MLflow experiment + writes `.env` (idempotent) |
| `scripts/deploy.py` | `uv run deploy`: loads `.env` so `BUNDLE_VAR_*` is set, then bundle validate + deploy + run (**Apps build**) |
| `scripts/log_model.py` | `uv run log-model`: logs `responses_agent.py` with `AuthPolicy(UserAuthPolicy+SystemAuthPolicy)` to MLflow + UC |
| `scripts/deploy_serving.py` | `uv run deploy-serving`: calls log-model then `databricks.agents.deploy()` (**Model Serving build**) |
| `scripts/quickstart.py` | Vendored upstream setup script (auth + experiment) — superseded for this repo by `setup-demo` / `deploy` |
| `scripts/discover_tools.py` | Discovers available workspace resources (vendored) |

---

## Agent Framework Capabilities

> **IMPORTANT:** When adding any tool to the agent, you MUST also grant permissions in `databricks.yml`. See the **add-tools** skill for required steps and examples.

**Tool Types:**
1. **Unity Catalog Function Tools** - SQL UDFs managed in UC with built-in governance
2. **Agent Code Tools** - Defined directly in agent code for REST APIs and low-latency operations
3. **MCP Tools** - Interoperable tools via Model Context Protocol (Databricks-managed, external, or self-hosted)

**Built-in Tools:**
- **system.ai.python_exec** - Execute Python code dynamically within agent queries (code interpreter)

**Common Patterns:**
- **Structured data retrieval** - Query SQL tables/databases
- **Unstructured data retrieval** - Document search and RAG via Vector Search
- **Code interpreter** - Python execution for analysis via system.ai.python_exec
- **External connections** - Integrate services like Slack via HTTP connections

Reference: https://docs.databricks.com/aws/en/generative-ai/agent-framework/
