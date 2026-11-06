"""Pinned external data downloads and scoring helpers for final-model training."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.request
from evaluation import possession_metrics as pm
from starter.sequence_possession import SAMPLE_ROOT, SAMPLE_REVISION
from starter.tracking_io import sha256

MODEL_SETTINGS = dict(max_iter=120, learning_rate=0.08, max_leaf_nodes=15,
                      min_samples_leaf=80, l2_regularization=5.0,
                      early_stopping=False, random_state=2026)

def download_samples(root=SAMPLE_ROOT):
    root = Path(root)
    names = ["README.md", "documentation/events-definitions.pdf"]
    for game in (1, 2):
        stem = "Sample_Game_{}".format(game)
        names.append("data/{}/{}_RawEventsData.csv".format(stem, stem))
        names += ["data/{}/{}_RawTrackingData_{}_Team.csv".format(stem, stem, side)
                  for side in ("Home", "Away")]

    def fetch(name):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/metrica-sports/sample-data/{}/{}".format(SAMPLE_REVISION, name)
        # Refresh atomically; failed downloads cannot leave a valid-looking CSV.
        temporary = destination.with_name(destination.name + ".download")
        try:
            with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return {"path": name, "url": url, "bytes": destination.stat().st_size, "sha256": sha256(destination)}

    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(fetch, names))
    (root / "manifest.json").write_text(json.dumps({
        "repository": "https://github.com/metrica-sports/sample-data", "commit": SAMPLE_REVISION,
        "files": records}, sort_keys=True, indent=2) + "\n")

def verify_samples(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["commit"] != SAMPLE_REVISION:
        raise ValueError("external sample revision differs from the pinned revision")
    for entry in manifest["files"]:
        if sha256(root / entry["path"]) != entry["sha256"]:
            raise ValueError("external sample hash mismatch: " + entry["path"])
    return manifest

def headline(reference, prediction):
    stats = pm.evaluate(reference, prediction, bootstrap_n=0)
    return {
        "macroF1": stats["macro_f1"], "spellF1": stats["spell_f1_mean"],
        "accuracy": stats["accuracy"], "dead_ball_F1": stats["per_class"]["D"]["f1"],
        "predicted_possessions": stats["n_pred_spells"], "reference_possessions": stats["n_gt_spells"],
        "graded_frames": stats["n_graded"],
    }
