"""Regenerate the Phase 2 dataset deterministically from a config file.

Usage:  python scripts/generate_dataset.py [--config configs/phase2.json] [--out artifacts/dataset]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schedx.features import build_feature_table  # noqa: E402
from schedx.workload import GeneratorConfig, config_dict, generate_dataset, to_frames  # noqa: E402


def dataset_version(cfg: dict) -> str:
    """Short hash of everything that determines the dataset's contents."""
    blob = json.dumps({"dataset": cfg["dataset"], "generator": cfg["generator"]}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase2.json"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "dataset"))
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    gen = GeneratorConfig.from_dict(cfg["generator"])
    ds = cfg["dataset"]
    workloads = generate_dataset(ds["n_workloads"], ds["master_seed"], gen)
    processes, wl = to_frames(workloads)
    features = build_feature_table(workloads, ds["horizons"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    processes.to_csv(out / "processes.csv", index=False)
    wl.to_csv(out / "workloads.csv", index=False)
    features.to_csv(out / "features.csv", index=False)
    meta = {
        "dataset_version": dataset_version(cfg),
        "config": {"dataset": ds, "generator": config_dict(gen)},
        "rows": {"workloads": len(wl), "processes": len(processes), "feature_rows": len(features)},
        "class_distribution": wl["workload_class"].value_counts().sort_index().to_dict(),
    }
    (out / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["rows"]), json.dumps(meta["class_distribution"]), f"-> {out}")


if __name__ == "__main__":
    main()
