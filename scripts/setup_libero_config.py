"""Create a noninteractive LIBERO config rooted in this checkout."""
from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "third_party" / "LIBERO" / "libero" / "libero"
CONFIG_ROOT = ROOT / ".cache" / "libero"
CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
config = {
    "benchmark_root": str(BENCHMARK_ROOT),
    "bddl_files": str(BENCHMARK_ROOT / "bddl_files"),
    "init_states": str(BENCHMARK_ROOT / "init_files"),
    "datasets": str(ROOT / "data" / "libero"),
    "assets": str(BENCHMARK_ROOT / "assets"),
}
for key, value in config.items():
    if key != "datasets":
        Path(value).mkdir(parents=True, exist_ok=True)
(CONFIG_ROOT / "config.yaml").write_text(
    yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
)
print(CONFIG_ROOT / "config.yaml")
print(yaml.safe_dump(config, sort_keys=True))
