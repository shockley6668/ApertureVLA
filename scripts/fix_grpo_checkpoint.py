"""Rewrite a GRPO checkpoint saved with text-cache-wrapped keys into the evaluator's key layout."""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_grpo import unwrap_state_keys  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoints", nargs="+")
for path in parser.parse_args().checkpoints:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    before = checkpoint["model_state_dict"]
    checkpoint["model_state_dict"] = unwrap_state_keys(before)
    renamed = sum(a != b for a, b in zip(before, checkpoint["model_state_dict"]))
    torch.save(checkpoint, path)
    print(f"{path}: renamed {renamed} keys")
