"""Log the IdentityEchoAgent to MLflow, register it to UC, and deploy it
as a Mosaic AI Agent serving endpoint.

Usage:
    DATABRICKS_CONFIG_PROFILE=<p> uv run python spikes/serving-obo/log_and_deploy.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import mlflow
from databricks import agents
from databricks.sdk import WorkspaceClient
from mlflow.models.auth_policy import AuthPolicy, SystemAuthPolicy, UserAuthPolicy
from mlflow.models.resources import DatabricksGenieSpace
from mlflow.types.responses import ResponsesAgentRequest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = Path(__file__).with_name("identity_echo.py")

UC_CATALOG = "serverless_stable_po64og_catalog"
UC_SCHEMA = "supervisor_obo_spike"
UC_MODEL_NAME = "identity_echo"
ENDPOINT_NAME = "supervisor-obo-spike"
PROBE_GENIE_SPACE_ID = "01f15474886a16d5aa027e0791fa855a"  # mirror identity_echo.py


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", help="Databricks CLI profile.")
    parser.add_argument(
        "--no-deploy",
        action="store_true",
        help="Log + register, but skip agents.deploy.",
    )
    args = parser.parse_args()

    if args.profile:
        os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    w = WorkspaceClient()
    me = w.current_user.me()
    print(f"==> Workspace: {w.config.host}  as {me.user_name}")

    # Create / get the spike experiment under the user's home so traces don't
    # pollute the main experiment.
    exp_path = f"/Users/{me.user_name}/supervisor-obo-spike"
    try:
        exp = w.experiments.get_by_name(experiment_name=exp_path)
        experiment_id = exp.experiment.experiment_id
    except Exception:
        exp = w.experiments.create_experiment(name=exp_path)
        experiment_id = exp.experiment_id
    print(f"==> Spike experiment: {exp_path}  id={experiment_id}")
    mlflow.set_experiment(experiment_id=experiment_id)

    # Smoke the model locally before logging.
    sys.path.insert(0, str(MODEL_FILE.parent))
    from identity_echo import IdentityEchoAgent  # noqa: E402

    local = IdentityEchoAgent()
    smoke = local.predict(
        ResponsesAgentRequest(input=[{"role": "user", "content": "ping"}])
    )
    print("==> Local smoke output type:", type(smoke).__name__)

    # OBO needs an AuthPolicy at log time:
    # - user_auth_policy.api_scopes = OAuth scopes minted into the user token
    #   that the serving runtime hands to ModelServingUserCredentials().
    # - system_auth_policy.resources = resources the endpoint SP is granted
    #   access to (for any code path that uses the default WorkspaceClient).
    auth_policy = AuthPolicy(
        user_auth_policy=UserAuthPolicy(api_scopes=["dashboards.genie", "sql"]),
        system_auth_policy=SystemAuthPolicy(
            resources=[DatabricksGenieSpace(genie_space_id=PROBE_GENIE_SPACE_ID)],
        ),
    )

    print("==> Logging model to MLflow…")
    with mlflow.start_run(run_name="identity-echo-spike"):
        info = mlflow.pyfunc.log_model(
            name="identity_echo",
            python_model=str(MODEL_FILE),
            pip_requirements=[
                "mlflow>=3.10.0",
                "databricks-agents>=1.9.3",
                "databricks-sdk>=0.40.0",
                "databricks-ai-bridge>=0.18.0",
            ],
            registered_model_name=f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}",
            auth_policy=auth_policy,
        )
    print(f"==> Logged model: {info.model_uri}")
    print(f"==> Registered model: {UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}")

    if args.no_deploy:
        print("--no-deploy set; stopping before agents.deploy()")
        return 0

    # Resolve version that was just registered.
    versions = w.registered_models.get(
        full_name=f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}"
    )
    latest_version = max(int(a.version) for a in versions.aliases or []) if False else None
    if latest_version is None:
        # Fall back to listing model versions.
        from databricks.sdk.service.catalog import ModelVersionInfo  # noqa: F401

        mvs = w.model_versions.list(
            full_name=f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}"
        )
        latest_version = max(int(mv.version) for mv in mvs)
    print(f"==> Latest UC model version: {latest_version}")

    print(f"==> Deploying as agent endpoint {ENDPOINT_NAME!r}…")
    deployment = agents.deploy(
        model_name=f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}",
        model_version=int(latest_version),
        endpoint_name=ENDPOINT_NAME,
        scale_to_zero=True,
        description="OBO spike — echoes whatever identity it can see inside the served model.",
    )
    print(f"==> Endpoint url: {deployment.query_endpoint}")
    print(f"==> Review app:   {deployment.review_app_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
