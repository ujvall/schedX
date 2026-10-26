"""Load a saved model and classify a newly generated workload.

Usage:  python scripts/predict.py [--model logistic_regression] [--seed 12345] [--index 0]

Defaults to the model selected on validation (reports/phase2/metrics.json).
The true label is shown only for comparison: it is computed from the full
traces, which a real scheduler would not have.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schedx.ml import load_model, predict_workload  # noqa: E402
from schedx.workload import GeneratorConfig, generate_workload  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="model name in artifacts/models (default: validation-selected)")
    ap.add_argument("--seed", type=int, default=12345, help="master seed for the new workload")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase2.json"))
    args = ap.parse_args()

    name = args.model or json.loads(
        (ROOT / "reports" / "phase2" / "metrics.json").read_text(encoding="utf-8")
    )["main_experiment"]["selected_model"]
    model, meta = load_model(ROOT / "artifacts" / "models" / f"{name}.joblib")
    if meta["dataset_version"] != json.loads(
        (ROOT / "artifacts" / "dataset" / "dataset_meta.json").read_text(encoding="utf-8")
    )["dataset_version"]:
        print("warning: model was trained on a different dataset version", file=sys.stderr)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    w = generate_workload(args.seed, args.index, GeneratorConfig.from_dict(cfg["generator"]))
    print(f"model {name} ({meta['model_type']}); workload {w.workload_id}: {len(w.processes)} processes")
    for k in cfg["dataset"]["horizons"]:
        out = predict_workload(model, meta, w.processes, k)
        probs = ", ".join(f"{c}={p:.2f}" for c, p in out["probabilities"].items())
        print(f"  k={k}: predicted {out['predicted']:<9} confidence {out['confidence']:.2f}  [{probs}]")
    print(f"  true label (retrospective, for comparison only): {w.label}")


if __name__ == "__main__":
    main()
