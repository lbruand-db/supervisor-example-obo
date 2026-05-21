"""Wrapper for `databricks bundle deploy && bundle run`.

Loads .env so BUNDLE_VAR_* keys (written by setup-demo) are picked up by
the Databricks CLI without the caller having to source the file. Then
runs `bundle validate`, `bundle deploy`, and `bundle run` in sequence.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_RESOURCE = "supervisor_example_obo"


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", "-p", help="Databricks CLI profile (sets DATABRICKS_CONFIG_PROFILE)."
    )
    parser.add_argument(
        "--target", "-t", default="dev", help="Bundle target (default: dev)."
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Deploy but don't restart the app.",
    )
    args = parser.parse_args()

    if not shutil.which("databricks"):
        print("ERROR: 'databricks' CLI not on PATH. Install it first.", file=sys.stderr)
        return 1

    load_dotenv(REPO_ROOT / ".env", override=False)
    required = [
        "BUNDLE_VAR_genie_finance_space_id",
        "BUNDLE_VAR_genie_sales_space_id",
        "BUNDLE_VAR_mlflow_experiment_id",
    ]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(
            "ERROR: missing required BUNDLE_VAR_* entries in .env:\n  "
            + "\n  ".join(missing)
            + "\n\nRun `uv run setup-demo --profile <p>` first to create the Genie\n"
            "spaces + MLflow experiment and populate .env.",
            file=sys.stderr,
        )
        return 1

    if args.profile:
        os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile

    _run(["databricks", "bundle", "validate", "-t", args.target])
    _run(["databricks", "bundle", "deploy", "-t", args.target])
    if not args.no_run:
        _run(["databricks", "bundle", "run", BUNDLE_RESOURCE, "-t", args.target])
    return 0


if __name__ == "__main__":
    sys.exit(main())
