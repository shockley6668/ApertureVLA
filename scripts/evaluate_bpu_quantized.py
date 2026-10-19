"""Compare a quantized HBIR model (quantized .bc, run with HBRuntime on x86) against float actions.

Runs inside the OE Docker image. Reads frames written by scripts/collect_bpu_frames.py and reports
per-split action error in the normalised action space plus gripper-sign agreement.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from horizon_tc_ui.hb_runtime import HBRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sess = HBRuntime(args.model)
    report = {"model": args.model, "input_names": list(sess.input_names), "output_names": list(sess.output_names)}
    for split in ("evaluation", "calibration"):
        folder = args.frames / split
        count = len(list((folder / "float_action").glob("*.npy")))
        abs_err, cosine, grip, ref_abs = [], [], [], []
        for i in range(count):
            feeds = {"camera_views": np.load(folder / "camera_views" / f"{i}.npy"),
                     "normalized_state": np.load(folder / "normalized_state" / f"{i}.npy")}
            out = np.asarray(sess.run(sess.output_names, feeds)[0], dtype=np.float32).reshape(12, 7)
            ref = np.load(folder / "float_action" / f"{i}.npy").reshape(12, 7)
            abs_err.append(np.abs(out - ref))
            ref_abs.append(np.abs(ref))
            cosine.append(float(np.dot(out.ravel(), ref.ravel()) / (np.linalg.norm(out) * np.linalg.norm(ref) + 1e-12)))
            grip.append(float(np.mean(np.sign(out[:, 6]) == np.sign(ref[:, 6]))))
        err = np.stack(abs_err)
        report[split] = {
            "frames": count,
            "mean_abs_error": float(err.mean()),
            "max_abs_error": float(err.max()),
            "mean_abs_error_per_dim": err.mean(axis=(0, 1)).round(5).tolist(),
            "mean_abs_float_action": float(np.stack(ref_abs).mean()),
            "cosine_min": float(min(cosine)),
            "cosine_mean": float(np.mean(cosine)),
            "gripper_sign_agreement": float(np.mean(grip)),
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
