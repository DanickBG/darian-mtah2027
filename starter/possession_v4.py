"""Generate v4 with robustness-trained ensemble and externally selected segment decoder."""

import argparse
import json
import os
from pathlib import Path
import platform
import sys

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import GAME, GAMES, half_video_blocks  # noqa: E402
from starter.tracking_io import write_submission  # noqa: E402
from starter.tracking_io import sha256  # noqa: E402
from starter.inference_features import competition_features, write_diagnostics  # noqa: E402
from starter.robust_possession import V4_ROOT, decode_duration  # noqa: E402
from starter.sequence_possession import LABELS, active_mask  # noqa: E402
from starter.train_possession_v4 import ensemble_probability  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/DaRiAn.joblib")
    parser.add_argument("--data-dir")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/results/possession_v4.csv")
    parser.add_argument("--diagnostics-dir", type=Path, default=V4_ROOT)
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()
    if not args.model.exists():
        parser.error("model missing; run python starter/train_possession_v4.py --download first")
    # The unchanged weights retain historical training source paths in their metadata.
    # The release manifest verifies the reorganised source used for inference.
    release = json.loads((ROOT / "DARIAN_CONFIG.json").read_text())
    if args.model.resolve() == (ROOT / release["model_relative_path"]).resolve():
        if sha256(args.model) != release["model_sha256"]:
            parser.error("packaged model hash differs from the release manifest")
    for name, expected in release["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            parser.error("release source changed: " + name)
    bundle = joblib.load(args.model)
    if sklearn.__version__ != bundle["metadata"]["sklearn"]:
        parser.error("scikit-learn differs from training; use requirements.txt")
    matrix, names, inputs = competition_features(args.diagnostics_dir, args.data_dir, args.rebuild_features)
    if names != bundle["feature_names"]:
        parser.error("feature schema differs from training")
    blocks, fps = half_video_blocks(GAME), GAMES[GAME]["fps"]
    active = active_mask(len(matrix), blocks)
    probabilities = np.full((len(matrix), 3), 1 / 3)
    probabilities[active] = ensemble_probability(bundle["models"], matrix[active], names)
    labels = decode_duration(probabilities, blocks, fps, **bundle["decoder"])
    write_submission(str(args.output), LABELS[labels])
    write_diagnostics(args.diagnostics_dir, matrix, names, probabilities, labels, fps, active)
    run = {
        "model_sha256": sha256(args.model), "decoder": bundle["decoder"],
        "input_and_feature_sources": inputs, "runner_sha256": sha256(Path(__file__)),
        "diagnostic_writer_sha256": sha256(ROOT / "starter/inference_features.py"),
        "python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
        "submission_sha256": sha256(args.output), "frames": len(labels),
        "active_label_counts": {LABELS[state]: int(np.sum(labels[active] == state)) for state in range(4)},
        "training_games": [1, 2], "reserved_evaluation_game": 3,
        "competition_labels_used": False,
        "note": "External proxy evaluation does not establish a winning competition score.",
    }
    (args.diagnostics_dir / "run.json").write_text(json.dumps(run, sort_keys=True, indent=2) + "\n")
    print("Wrote {} ({} frames).".format(args.output, len(labels)))
    print("Local probabilities, transitions and provenance: {}".format(args.diagnostics_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
