"""Apply the prespecified early-compression latency gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-report", type=Path, required=True)
    parser.add_argument("--early-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", default="host_rgb_to_env_action_chunk")
    parser.add_argument("--minimum-improvement", type=float, default=0.10)
    args = parser.parse_args()
    if not 0 < args.minimum_improvement < 1:
        parser.error("minimum improvement must be between zero and one")

    dense = json.loads(args.dense_report.read_text(encoding="utf-8"))
    early = json.loads(args.early_report.read_text(encoding="utf-8"))
    dense_p95 = float(dense["metrics"][args.metric]["p95_ms"])
    early_p95 = float(early["metrics"][args.metric]["p95_ms"])
    improvement = 1.0 - early_p95 / dense_p95
    report = {
        "evidence": "early_compression_full_input_speed_gate_v1",
        "metric": args.metric,
        "dense_report": str(args.dense_report),
        "early_report": str(args.early_report),
        "dense_p95_ms": dense_p95,
        "early_p95_ms": early_p95,
        "p95_improvement_fraction": improvement,
        "p95_speedup": dense_p95 / early_p95,
        "minimum_improvement_fraction": args.minimum_improvement,
        "passed": improvement >= args.minimum_improvement,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
