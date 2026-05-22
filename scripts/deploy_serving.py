"""Log + register the model, then deploy it as a Mosaic AI agent endpoint.

Sister script to `scripts/deploy.py` (which deploys the Apps build).
Both read `.env`; this one calls `scripts/log_model:main` to log + register
the model, then `databricks.agents.deploy()` to create or update the
serving endpoint with the Genie / LLM environment vars wired in.

Usage:
    uv run deploy-serving --profile <p>
    uv run deploy-serving --profile <p> --endpoint-name my-endpoint

Prereq: the workspace `Agent Framework: On-Behalf-Of-User Authorization`
preview must be enabled, otherwise OBO calls fail with a runtime
`ValueError`. See SPECS/PLAN_MODEL_SERVING.md §1.1.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from databricks import agents
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ENDPOINT = "supervisor-example-obo-serving"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", "-p", help="Databricks CLI profile.")
    parser.add_argument("--endpoint-name", help="Override the serving endpoint name.")
    parser.add_argument(
        "--no-scale-to-zero",
        action="store_true",
        help="Keep at least one replica warm (default: scale to zero).",
    )
    parser.add_argument(
        "--uc-catalog", help="UC catalog for the registered model (passed to log-model)."
    )
    parser.add_argument("--uc-schema", help="UC schema for the registered model.")
    parser.add_argument("--uc-model-name", help="UC model name.")
    args = parser.parse_args()

    if args.profile:
        os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile

    load_dotenv(REPO_ROOT / ".env", override=False)

    # Log + register the model first. We call scripts.log_model:main rather
    # than spawning a subprocess so the in-process env (profile, dotenv)
    # already applies and we share auth.
    from scripts import log_model

    log_argv = []
    if args.uc_catalog:
        log_argv += ["--uc-catalog", args.uc_catalog]
    if args.uc_schema:
        log_argv += ["--uc-schema", args.uc_schema]
    if args.uc_model_name:
        log_argv += ["--uc-model-name", args.uc_model_name]
    _saved_argv = sys.argv
    sys.argv = ["log-model", *log_argv]
    try:
        rc = log_model.main()
    finally:
        sys.argv = _saved_argv
    if rc != 0:
        return rc

    catalog = args.uc_catalog or os.environ.get("SERVING_UC_CATALOG", log_model.DEFAULT_CATALOG)
    schema = args.uc_schema or os.environ.get("SERVING_UC_SCHEMA", log_model.DEFAULT_SCHEMA)
    name = args.uc_model_name or os.environ.get(
        "SERVING_UC_MODEL_NAME", log_model.DEFAULT_MODEL_NAME
    )
    full_model_name = f"{catalog}.{schema}.{name}"

    endpoint_name = (
        args.endpoint_name or os.environ.get("SERVING_ENDPOINT_NAME", DEFAULT_ENDPOINT)
    )

    w = WorkspaceClient()
    latest = max(int(mv.version) for mv in w.model_versions.list(full_name=full_model_name))
    print(f"==> Deploying {full_model_name} v{latest} to endpoint {endpoint_name!r}…")

    dep = agents.deploy(
        model_name=full_model_name,
        model_version=int(latest),
        endpoint_name=endpoint_name,
        scale_to_zero=not args.no_scale_to_zero,
        description="Hierarchical L1->L2->Genie supervisor on Mosaic AI Model Serving with OBO.",
        environment_vars={
            "GENIE_FINANCE_SPACE_ID": os.environ["GENIE_FINANCE_SPACE_ID"],
            "GENIE_SALES_SPACE_ID": os.environ["GENIE_SALES_SPACE_ID"],
            "LLM_ENDPOINT": os.environ.get("LLM_ENDPOINT", "databricks-gpt-5-2"),
        },
    )
    print(f"==> Endpoint query URL: {dep.query_endpoint}")
    print(f"==> Review app:         {dep.review_app_url}")
    print()
    print("First deploy provisions compute and takes ~5-15 min. Tail status with:")
    print(f"  databricks serving-endpoints get {endpoint_name} --profile {args.profile or '<p>'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
