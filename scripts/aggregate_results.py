"""Aggregate LIBERO evaluation JSON reports into one summary table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ("task_suite_name", "total_episodes", "total_successes", "overall_success_rate")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{path} is missing keys: {missing}")
    return {
        "file": path.name,
        "suite": payload["task_suite_name"],
        "episodes": int(payload["total_episodes"]),
        "successes": int(payload["total_successes"]),
        "success_rate": float(payload["overall_success_rate"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = [load_result(path) for path in args.reports]
    rows.sort(key=lambda row: (row["suite"], row["file"]))
    payload = {
        "report_count": len(rows),
        "total_episodes": sum(row["episodes"] for row in rows),
        "total_successes": sum(row["successes"] for row in rows),
        "overall_success_rate": (
            sum(row["successes"] for row in rows)
            / sum(row["episodes"] for row in rows)
        ),
        "results": rows,
    }
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
