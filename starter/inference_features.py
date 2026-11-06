"""Cached tracking features and local diagnostics for the final model."""

import csv
import json
from pathlib import Path
import numpy as np
from games import require
from starter.tracking_io import sha256
from starter.sequence_possession import LABELS, competition_tracking, feature_matrix

ROOT = Path(__file__).resolve().parents[1]

def competition_features(folder, data_dir=None, rebuild=False):
    paths = require(["atd_meta", "atd_raw"], data_dir=data_dir)
    sources = [ROOT / "starter/sequence_possession.py", ROOT / "starter/tracking_io.py",
               ROOT / "starter/inference_features.py", ROOT / "games.py"]
    key = {
        "inputs": {Path(path).name: sha256(path) for path in paths},
        "sources": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "numpy": np.__version__,
    }
    cache = folder / "competition_features.npz"
    if cache.exists() and not rebuild:
        with np.load(cache, allow_pickle=False) as values:
            if str(values["key"].item()) == json.dumps(key, sort_keys=True):
                print("Using input/source-verified local feature cache.", flush=True)
                return values["features"].copy(), values["names"].tolist(), key
    print("Reading tactical tracking and player movement; no video or ground truth loaded.", flush=True)
    tracking = competition_tracking(data_dir)
    matrix, names = feature_matrix(tracking)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, key=json.dumps(key, sort_keys=True), features=matrix, names=np.array(names))
    return matrix, names, key

def write_diagnostics(folder, matrix, names, probabilities, labels, fps, mask):
    folder.mkdir(parents=True, exist_ok=True)
    raw = np.argmax(probabilities, axis=1)
    reasons = np.full(len(labels), "outside_half", dtype="<U40")
    reasons[mask & (raw == labels)] = "strongest_frame_evidence"
    reasons[mask & (raw != labels)] = "temporal_context_resolved"
    with (folder / "frames.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["frame", "home_probability", "away_probability", "dead_probability",
                         "state", "reason", "ball_visible", "previous_ball_age_s", "next_ball_age_s",
                         "home_distance_m", "away_distance_m", "player_speed_median"])
        indices = [names.index(name) for name in ("ball_visible", "previous_ball_age", "next_ball_age",
                                                  "home_distance", "away_distance", "player_speed_median")]
        for frame, label in enumerate(labels):
            features = matrix[frame, indices].copy()
            features[1:3] = np.expm1(features[1:3])
            writer.writerow([frame] + ["{:.6f}".format(p) if mask[frame] else "" for p in probabilities[frame]]
                            + [LABELS[label], reasons[frame]]
                            + ["{:.4f}".format(value) if np.isfinite(value) else "" for value in features])
    with (folder / "transitions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["frame", "video_seconds", "from", "to", "reason"])
        for frame in np.flatnonzero(labels[1:] != labels[:-1]) + 1:
            writer.writerow([frame, "{:.3f}".format(frame / fps), LABELS[labels[frame - 1]],
                             LABELS[labels[frame]], reasons[frame]])
