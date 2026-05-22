"""Log `agent_server/responses_agent.py` to MLflow + register it to UC.

Reads from .env:

  GENIE_FINANCE_SPACE_ID, GENIE_SALES_SPACE_ID     (required)
  LLM_ENDPOINT                                      (default: databricks-gpt-5-2)
  MLFLOW_EXPERIMENT_ID                              (optional; defaults to
                                                     /Users/<me>/supervisor-example-obo)
  SERVING_UC_CATALOG, SERVING_UC_SCHEMA, SERVING_UC_MODEL_NAME
                                                    (with sensible defaults)

Use `uv run setup-demo` first to populate GENIE_*_SPACE_ID.

Usage:
  uv run log-model --profile <p>
  uv run log-model --profile <p> --uc-catalog mycat --uc-schema myschema
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from dotenv import load_dotenv
from mlflow.models.auth_policy import AuthPolicy, SystemAuthPolicy, UserAuthPolicy
from mlflow.models.resources import DatabricksGenieSpace, DatabricksServingEndpoint

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = REPO_ROOT / "agent_server" / "responses_agent.py"
AGENT_SERVER_PKG = REPO_ROOT / "agent_server"

DEFAULT_CATALOG = "main"
DEFAULT_SCHEMA = "supervisor_example_obo"
DEFAULT_MODEL_NAME = "supervisor"


def _info(msg: str) -> None:
    print(f"==> {msg}")


def _ensure_schema(w: WorkspaceClient, catalog: str, schema: str) -> None:
    try:
        w.schemas.get(full_name=f"{catalog}.{schema}")
        return
    except NotFound:
        pass
    w.schemas.create(name=schema, catalog_name=catalog)
    _info(f"Created UC schema {catalog}.{schema}")


def _resolve_experiment(w: WorkspaceClient, env_id: str | None) -> str:
    if env_id:
        return env_id
    me = w.current_user.me().user_name
    exp_path = f"/Users/{me}/supervisor-example-obo"
    try:
        exp = w.experiments.get_by_name(experiment_name=exp_path)
        return exp.experiment.experiment_id
    except Exception:
        created = w.experiments.create_experiment(name=exp_path)
        return created.experiment_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", "-p", help="Databricks CLI profile.")
    parser.add_argument("--uc-catalog", help="UC catalog for the registered model.")
    parser.add_argument("--uc-schema", help="UC schema for the registered model.")
    parser.add_argument("--uc-model-name", help="UC model name (no catalog/schema).")
    args = parser.parse_args()

    if args.profile:
        os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile

    load_dotenv(REPO_ROOT / ".env", override=False)

    try:
        finance_id = os.environ["GENIE_FINANCE_SPACE_ID"]
        sales_id = os.environ["GENIE_SALES_SPACE_ID"]
    except KeyError as e:
        print(
            f"ERROR: {e.args[0]} not set in env / .env. Run `uv run setup-demo` first.",
            file=sys.stderr,
        )
        return 1

    llm_endpoint = os.environ.get("LLM_ENDPOINT", "databricks-gpt-5-2")
    catalog = args.uc_catalog or os.environ.get("SERVING_UC_CATALOG", DEFAULT_CATALOG)
    schema = args.uc_schema or os.environ.get("SERVING_UC_SCHEMA", DEFAULT_SCHEMA)
    model_name = (
        args.uc_model_name or os.environ.get("SERVING_UC_MODEL_NAME", DEFAULT_MODEL_NAME)
    )
    full_model_name = f"{catalog}.{schema}.{model_name}"

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    w = WorkspaceClient()
    me = w.current_user.me().user_name
    _info(f"Workspace: {w.config.host}  as {me}")

    _ensure_schema(w, catalog, schema)

    experiment_id = _resolve_experiment(w, os.environ.get("MLFLOW_EXPERIMENT_ID"))
    mlflow.set_experiment(experiment_id=experiment_id)
    _info(f"MLflow experiment id={experiment_id}")

    auth_policy = AuthPolicy(
        user_auth_policy=UserAuthPolicy(api_scopes=["dashboards.genie", "sql"]),
        system_auth_policy=SystemAuthPolicy(
            resources=[
                DatabricksGenieSpace(genie_space_id=finance_id),
                DatabricksGenieSpace(genie_space_id=sales_id),
                DatabricksServingEndpoint(endpoint_name=llm_endpoint),
            ],
        ),
    )

    _info(f"Logging model to MLflow as {full_model_name}…")
    with mlflow.start_run(run_name="supervisor-serving-deploy"):
        info = mlflow.pyfunc.log_model(
            name=model_name,
            python_model=str(MODEL_FILE),
            code_paths=[str(AGENT_SERVER_PKG)],
            pip_requirements=[
                "mlflow>=3.10.0",
                "databricks-agents>=1.9.3",
                "databricks-sdk>=0.40.0",
                "databricks-ai-bridge>=0.18.0",
                "databricks-langchain>=0.17.0",
                "langchain>=1.0.0",
                "langgraph>=1.1.0",
            ],
            registered_model_name=full_model_name,
            auth_policy=auth_policy,
        )
    _info(f"Logged: {info.model_uri}")

    mvs = list(w.model_versions.list(full_name=full_model_name))
    latest = max(int(mv.version) for mv in mvs)
    _info(f"Latest UC version: {latest}")
    print(latest)  # final stdout line = the version, so deploy_serving can capture it
    return 0


if __name__ == "__main__":
    sys.exit(main())
