"""Bootstrap the Databricks-side demo resources for supervisor-example-obo.

What it does (idempotent: skip if the equivalent resource already exists):

  1. Pick (or create) an MLflow experiment under the calling user's home
     workspace path.
  2. Create two Genie spaces — one over samples.tpch.* (finance) and one
     over samples.bakehouse.* (sales) — via `/api/2.0/data-rooms`. The
     public Genie SDK has no `create_space`, hence the older endpoint.
  3. Pick a serverless SQL warehouse from the workspace (the Genie space
     needs one to drive the SQL queries).
  4. Write the IDs into `.env` as both:
       • plain dev keys (MLFLOW_EXPERIMENT_ID, GENIE_*_SPACE_ID) — used
         locally by `start-server` / `start-app`.
       • BUNDLE_VAR_* keys — picked up by `databricks bundle deploy`
         (via `uv run deploy`, which loads .env first).

Re-run safely: if the .env already has IDs, the script reuses them.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from databricks.sdk import WorkspaceClient
from dotenv import set_key

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"
ENV_EXAMPLE = REPO_ROOT / ".env.example"

FINANCE_TABLES = [
    "samples.tpch.orders",
    "samples.tpch.lineitem",
    "samples.tpch.customer",
    "samples.tpch.nation",
    "samples.tpch.region",
]
SALES_TABLES = [
    "samples.bakehouse.sales_transactions",
    "samples.bakehouse.sales_customers",
    "samples.bakehouse.sales_franchises",
    "samples.bakehouse.sales_suppliers",
]


def _info(msg: str) -> None:
    print(f"==> {msg}")


def _warn(msg: str) -> None:
    print(f"--  {msg}")


def _ensure_env_file() -> None:
    if ENV_PATH.exists():
        return
    if ENV_EXAMPLE.exists():
        ENV_PATH.write_text(ENV_EXAMPLE.read_text())
        _info(f"Created {ENV_PATH.name} from .env.example")
    else:
        ENV_PATH.touch()
        _info(f"Created empty {ENV_PATH.name}")


def _read_env_var(key: str) -> str | None:
    if not ENV_PATH.exists():
        return None
    for raw in ENV_PATH.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == key:
            return v.strip().strip('"').strip("'") or None
    return None


def _write_env_vars(pairs: dict[str, str]) -> None:
    """Set or update KEY=VALUE entries in .env, preserving the rest."""
    for k, v in pairs.items():
        set_key(str(ENV_PATH), k, v, quote_mode="never")


def _pick_warehouse(w: WorkspaceClient, prefer_serverless: bool = True) -> str:
    """Return a warehouse_id. Prefer serverless + non-stopped."""
    warehouses = list(w.warehouses.list())
    if not warehouses:
        raise RuntimeError(
            "No SQL warehouses in the workspace. Create one before running setup-demo."
        )

    def score(wh) -> tuple[int, int]:
        # Higher is better. Serverless first, running first.
        s = (1 if prefer_serverless and wh.enable_serverless_compute else 0) * 10
        s += 1 if (wh.state and "RUNNING" in str(wh.state).upper()) else 0
        return (s, 0)

    best = max(warehouses, key=score)
    _info(f"Using SQL warehouse: {best.name!r}  id={best.id}  state={best.state}")
    return best.id


def _ensure_experiment(w: WorkspaceClient, name: str | None) -> str:
    """Pick the named experiment if it exists, else create it."""
    me = w.current_user.me().user_name
    exp_name = name or f"/Users/{me}/supervisor-example-obo"

    try:
        existing = w.experiments.get_by_name(experiment_name=exp_name)
        eid = existing.experiment.experiment_id
        _info(f"Re-using MLflow experiment {exp_name!r}  id={eid}")
        return eid
    except Exception:
        pass

    created = w.experiments.create_experiment(name=exp_name)
    _info(f"Created MLflow experiment {exp_name!r}  id={created.experiment_id}")
    return created.experiment_id


def _create_genie_space(
    w: WorkspaceClient,
    *,
    display_name: str,
    description: str,
    warehouse_id: str,
    table_identifiers: list[str],
) -> str:
    """Create a Genie space via /api/2.0/data-rooms. Returns the space_id."""
    resp = w.api_client.do(
        "POST",
        "/api/2.0/data-rooms",
        body={
            "display_name": display_name,
            "description": description,
            "warehouse_id": warehouse_id,
            "table_identifiers": table_identifiers,
        },
    )
    space_id = resp["space_id"]
    _info(f"Created Genie space {display_name!r}  id={space_id}")
    return space_id


def _maybe_get_space(w: WorkspaceClient, space_id: str | None) -> str | None:
    """If `space_id` is set and the space still exists, return it; else None."""
    if not space_id:
        return None
    try:
        sp = w.genie.get_space(space_id)
        _info(f"Re-using existing Genie space {sp.title!r}  id={space_id}")
        return space_id
    except Exception:
        _warn(f"Stored Genie space id {space_id} no longer resolves; will recreate.")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        "-p",
        help="Databricks CLI profile (sets DATABRICKS_CONFIG_PROFILE).",
    )
    parser.add_argument(
        "--experiment-path",
        help="MLflow experiment path. Defaults to /Users/<me>/supervisor-example-obo.",
    )
    parser.add_argument(
        "--warehouse-id",
        help="SQL warehouse id to back the Genie spaces. If omitted, picks the first serverless warehouse.",
    )
    args = parser.parse_args()

    if args.profile:
        os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile

    _ensure_env_file()
    w = WorkspaceClient()
    me = w.current_user.me().user_name
    _info(f"Workspace: {w.config.host}  as {me}")

    # 1. Experiment
    experiment_id = _read_env_var("MLFLOW_EXPERIMENT_ID")
    if experiment_id:
        _info(f"Re-using MLflow experiment from .env  id={experiment_id}")
    else:
        experiment_id = _ensure_experiment(w, args.experiment_path)

    # 2. Warehouse
    warehouse_id = args.warehouse_id or _pick_warehouse(w)

    # 3. Genie spaces
    finance_id = _maybe_get_space(w, _read_env_var("GENIE_FINANCE_SPACE_ID"))
    if not finance_id:
        finance_id = _create_genie_space(
            w,
            display_name="Supervisor OBO - Finance",
            description=(
                "Finance Genie space for supervisor-example-obo. Tables: "
                "samples.tpch.* (treat orders/lineitem as revenue data)."
            ),
            warehouse_id=warehouse_id,
            table_identifiers=FINANCE_TABLES,
        )

    sales_id = _maybe_get_space(w, _read_env_var("GENIE_SALES_SPACE_ID"))
    if not sales_id:
        sales_id = _create_genie_space(
            w,
            display_name="Supervisor OBO - Sales",
            description=(
                "Sales Genie space for supervisor-example-obo. Tables: "
                "samples.bakehouse.sales_*."
            ),
            warehouse_id=warehouse_id,
            table_identifiers=SALES_TABLES,
        )

    # 4. Persist into .env (both dev keys and BUNDLE_VAR_* keys).
    pairs = {
        "MLFLOW_EXPERIMENT_ID": experiment_id,
        "GENIE_FINANCE_SPACE_ID": finance_id,
        "GENIE_SALES_SPACE_ID": sales_id,
        "BUNDLE_VAR_mlflow_experiment_id": experiment_id,
        "BUNDLE_VAR_genie_finance_space_id": finance_id,
        "BUNDLE_VAR_genie_sales_space_id": sales_id,
    }
    _write_env_vars(pairs)
    _info(f"Wrote IDs into {ENV_PATH.name}")

    print()
    print("Next steps:")
    print("  uv run deploy --profile <your-profile>   # deploys + starts the app")
    print("  # or test locally first:")
    print("  uv run start-server   # localhost:8000, OBO_FALLBACK_TO_DEFAULT=1 picks up your CLI auth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
