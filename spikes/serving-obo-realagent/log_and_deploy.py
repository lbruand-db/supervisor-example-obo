"""Log the real-agent SupervisorAgent (the same L1→L2→Genie graph that the
Apps build runs) as an MLflow model, register to UC, and deploy as a
Mosaic AI agent serving endpoint with OBO.

Usage:
    DATABRICKS_CONFIG_PROFILE=<p> uv run python spikes/serving-obo-realagent/log_and_deploy.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import mlflow
from databricks import agents
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
from mlflow.models.auth_policy import AuthPolicy, SystemAuthPolicy, UserAuthPolicy
from mlflow.models.resources import DatabricksGenieSpace, DatabricksServingEndpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = Path(__file__).with_name("supervisor_responses_agent.py")
AGENT_SERVER_PKG = REPO_ROOT / "agent_server"

UC_CATALOG = "serverless_stable_po64og_catalog"
UC_SCHEMA = "supervisor_obo_realagent"
UC_MODEL_NAME = "supervisor"
ENDPOINT_NAME = "supervisor-example-obo-serving"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", help="Databricks CLI profile.")
    parser.add_argument(
        "--no-deploy", action="store_true", help="Log + register, skip agents.deploy."
    )
    args = parser.parse_args()

    if args.profile:
        os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile

    # Pick up GENIE_*_SPACE_ID / LLM_ENDPOINT / MLFLOW_EXPERIMENT_ID from .env
    load_dotenv(REPO_ROOT / ".env", override=False)

    finance_id = os.environ["GENIE_FINANCE_SPACE_ID"]
    sales_id = os.environ["GENIE_SALES_SPACE_ID"]
    llm_endpoint = os.environ.get("LLM_ENDPOINT", "databricks-gpt-5-2")

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    w = WorkspaceClient()
    me = w.current_user.me().user_name
    print(f"==> Workspace: {w.config.host}  as {me}")

    exp_path = f"/Users/{me}/supervisor-obo-realagent-spike"
    try:
        exp = w.experiments.get_by_name(experiment_name=exp_path)
        experiment_id = exp.experiment.experiment_id
    except Exception:
        exp = w.experiments.create_experiment(name=exp_path)
        experiment_id = exp.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    print(f"==> Experiment: {exp_path}  id={experiment_id}")

    # Ensure UC schema exists.
    try:
        w.schemas.create(name=UC_SCHEMA, catalog_name=UC_CATALOG)
        print(f"==> Created UC schema {UC_CATALOG}.{UC_SCHEMA}")
    except Exception:
        pass  # already exists

    # Local smoke before logging — just imports cleanly. Don't run predict()
    # here: ModelServingUserCredentials only resolves inside a real serving
    # request.
    sys.path.insert(0, str(MODEL_FILE.parent))
    sys.path.insert(0, str(REPO_ROOT))
    from supervisor_responses_agent import SupervisorAgent  # noqa: F401

    print("==> Local import smoke OK")

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

    print("==> Logging model to MLflow…")
    with mlflow.start_run(run_name="supervisor-realagent-spike"):
        info = mlflow.pyfunc.log_model(
            name="supervisor",
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
            registered_model_name=f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}",
            auth_policy=auth_policy,
        )
    print(f"==> Logged: {info.model_uri}")

    if args.no_deploy:
        return 0

    mvs = w.model_versions.list(full_name=f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}")
    latest = max(int(mv.version) for mv in mvs)
    print(f"==> Latest version: {latest}")

    print(f"==> Deploying to endpoint {ENDPOINT_NAME!r}…")
    dep = agents.deploy(
        model_name=f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}",
        model_version=int(latest),
        endpoint_name=ENDPOINT_NAME,
        scale_to_zero=True,
        description="Real-agent spike: hierarchical supervisor on Mosaic AI Model Serving with OBO.",
        environment_vars={
            "GENIE_FINANCE_SPACE_ID": finance_id,
            "GENIE_SALES_SPACE_ID": sales_id,
            "LLM_ENDPOINT": llm_endpoint,
        },
    )
    print(f"==> Endpoint url: {dep.query_endpoint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
