"""Develop on public Games 1/2 with game-wise splits; reserve Game 3 for evaluation.

No competition data, public competition labels, or private labels are opened.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import platform
import sys
import urllib.request

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm  # noqa: E402
from starter.tracking_io import sha256  # noqa: E402
from starter.robust_possession import (  # noqa: E402
    PROFILES, THIRD_ROOT, V4_ROOT, decode_duration, perturb, rich_sample, third_reference,
)
from starter.sequence_possession import (  # noqa: E402
    SAMPLE_REVISION, SAMPLE_ROOT, emission_probabilities, feature_matrix,
    mirrored_features, sample_reference,
)
from starter.training_data import (  # noqa: E402
    MODEL_SETTINGS, download_samples, headline, verify_samples,
)

SWITCH_COSTS = (0.25, 0.5, 1.0, 2.0)
MINIMUM_SECONDS = (0.0, 0.4, 0.8)
DECODE_STEP = 5
SPELLF1_TARGET = 0.49


def select_decoder_config(candidates):
    """Prefer the highest spellF1 subject to the challenge floor and worst-case robustness.

    The fundraiser metric is the mean possession-spell F1 over IoU 0.3..0.7. Higher is better
    because `headline()` records `stats["spell_f1_mean"]` and the board ranks on that quantity.
    We therefore enforce a minimum floor before using the historical development objective.
    """
    feasible = [
        row for row in candidates
        if row["mean_spellF1"] >= SPELLF1_TARGET and row["worst_profile_spellF1"] >= SPELLF1_TARGET
    ]
    pool = feasible or candidates
    return max(pool, key=lambda row: (row["selection_objective"], row["mean_spellF1"],
                                      -row["minimum_s"], -row["switch_cost"]))


def download_third(root=THIRD_ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    names = ["Sample_Game_3_metadata.xml", "Sample_Game_3_tracking.txt", "Sample_Game_3_events.json"]
    def fetch(name):
        path = root / name
        temporary = path.with_name(path.name + ".download")
        url = "https://raw.githubusercontent.com/metrica-sports/sample-data/{}/data/Sample_Game_3/{}".format(SAMPLE_REVISION, name)
        try:
            with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return {"name": name, "url": url, "sha256": sha256(path), "bytes": path.stat().st_size}
    with ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(fetch, names))
    (root / "manifest.json").write_text(json.dumps({"commit": SAMPLE_REVISION, "files": files}, indent=2) + "\n")


def verify_third(root):
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    if manifest["commit"] != SAMPLE_REVISION:
        raise ValueError("reserved sample revision differs")
    for entry in manifest["files"]:
        if sha256(Path(root) / entry["name"]) != entry["sha256"]:
            raise ValueError("reserved sample hash mismatch")
    return manifest


def fit_robust(tracking, reference, seed):
    mask = (reference != 3) & (np.arange(len(reference)) % 5 == 0)
    for frame in np.flatnonzero(reference[1:] != reference[:-1]) + 1:
        mask[max(0, frame - 5):frame + 6] = False
    x, y, names = [], [], None
    for index, profile in enumerate(PROFILES):
        noisy = perturb(tracking, seed + index, profile)
        features, names = feature_matrix(noisy)
        selected, labels = features[mask], reference[mask]
        mirrored = labels.copy()
        mirrored[labels == 0], mirrored[labels == 1] = 1, 0
        x.extend((selected, mirrored_features(selected, names)))
        y.extend((labels, mirrored))
    features, labels = np.vstack(x), np.concatenate(y)
    if set(labels) != {0, 1, 2}:
        raise ValueError("all three classes are required for training")
    model = HistGradientBoostingClassifier(**MODEL_SETTINGS)
    with threadpool_limits(limits=1):
        model.fit(features, labels)
    return model, names, len(labels)


def ensemble_probability(models, features, names):
    with threadpool_limits(limits=1):
        return np.mean([emission_probabilities(model, features, names) for model in models], axis=0)




def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--download", action="store_true", help="download the three pinned public samples (~185 MB)")
    parser.add_argument("--samples", type=Path, default=SAMPLE_ROOT)
    parser.add_argument("--third-sample", type=Path, default=THIRD_ROOT)
    parser.add_argument("--output-dir", type=Path, default=V4_ROOT)
    args = parser.parse_args()
    if args.download:
        download_samples(args.samples)
        download_third(args.third_sample)
    manifest = verify_samples(args.samples)
    reserved_manifest = verify_third(args.third_sample)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tracks, references, models, development, training_rows = {}, {}, [], [], []
    for game in (1, 2):
        print("Loading development Game {}; no competition labels used.".format(game), flush=True)
        tracks[game] = rich_sample(game, args.samples, args.third_sample, args.output_dir / "cache")
        references[game] = sample_reference(game, tracks[game], args.samples)
    for training_game, validation_game in ((1, 2), (2, 1)):
        print("Fitting robust model on Game {}; validating on Game {}.".format(training_game, validation_game), flush=True)
        model, names, rows = fit_robust(tracks[training_game], references[training_game], 1000 + 10 * training_game)
        models.append(model)
        training_rows.append(rows)
        for index, profile in enumerate(PROFILES):
            tracking = perturb(tracks[validation_game], 4000 + 10 * validation_game + index, profile)
            features, actual_names = feature_matrix(tracking)
            if actual_names != names:
                raise ValueError("feature schema mismatch")
            probabilities = ensemble_probability([model], features, names)
            development.append((validation_game, profile, probabilities))
        print("Completed fold {} -> {}.".format(training_game, validation_game), flush=True)
    candidates = []
    for minimum in MINIMUM_SECONDS:
        for cost in SWITCH_COSTS:
            scores = []
            for game, profile, probabilities in development:
                labels = decode_duration(probabilities, tracks[game].blocks, tracks[game].fps,
                                         cost, minimum, DECODE_STEP)
                score = headline(references[game], labels)
                scores.append({"game": game, "profile": profile, **score})
            profile_means = {profile: float(np.mean([row["spellF1"] for row in scores if row["profile"] == profile]))
                             for profile in PROFILES}
            mean_score = float(np.mean([row["spellF1"] for row in scores]))
            worst_profile = min(profile_means.values())
            objective = 0.5 * mean_score + 0.5 * worst_profile
            candidate = {"switch_cost": cost, "minimum_s": minimum, "step": DECODE_STEP,
                         "mean_spellF1": mean_score, "worst_profile_spellF1": worst_profile,
                         "selection_objective": objective, "scores": scores}
            candidates.append(candidate)
            print("Game-wise development: minimum {:.1f}s, cost {:.2f}, mean {:.4f}, worst {:.4f}.".format(
                minimum, cost, mean_score, worst_profile), flush=True)
    selected = select_decoder_config(candidates)
    settings = {name: selected[name] for name in ("switch_cost", "minimum_s", "step")}
    sources = [Path(__file__), ROOT / "starter/robust_possession.py", ROOT / "starter/sequence_possession.py",
               ROOT / "starter/training_data.py", ROOT / "starter/tracking_io.py", ROOT / "games.py"]
    metadata = {
        "training_games": [1, 2], "reserved_evaluation_game": 3,
        "development_split": "two leave-one-game-out folds; Game 3 excluded from selection",
        "reference_type": "other-match event-derived proxy, not corrected frame-level possession GT",
        "source": manifest, "reserved_source": reserved_manifest,
        "model_settings": MODEL_SETTINGS, "training_rows_per_model": training_rows,
        "feature_names": names, "decoder": settings, "development_candidates": candidates,
        "selection_objective": "half mean spellF1 + half worst-profile mean spellF1, over game-wise folds",
        "python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
        "sources_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "competition_labels_used": False,
    }
    bundle = {"models": models, "feature_names": names, "decoder": settings, "metadata": metadata}
    joblib.dump(bundle, args.output_dir / "model.joblib", compress=0)
    (args.output_dir / "training.json").write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    print("Frozen settings: {}. Opening reserved Game 3 event labels now.".format(settings), flush=True)
    third = rich_sample(3, args.samples, args.third_sample, args.output_dir / "cache")
    reference = third_reference(third, args.third_sample)
    report = {
        "reserved_game": 3, "reference_type": metadata["reference_type"],
        "decoder": settings, "v4_model_sha256": sha256(args.output_dir / "model.joblib"),
        "conditions": {},
        "limitations": ["Only one reserved match", "Event-proxy labels are approximate",
                        "Synthetic tracking errors are assumptions", "No competition ranking guarantee"],
    }
    for index, profile in enumerate(PROFILES):
        tracking = perturb(third, 9000 + index, profile)
        features, schema = feature_matrix(tracking)
        probabilities = ensemble_probability(models, features, schema)
        v4_labels = decode_duration(probabilities, tracking.blocks, tracking.fps, **settings)
        result = {"possession_v4": headline(reference, v4_labels)}
        report["conditions"][profile] = result
        print("Reserved OTHER match, {}:".format(profile), flush=True)
        for name, score in result.items():
            print("  {:20s} macroF1 {:.4f}, spellF1 {:.4f}".format(name, score["macroF1"], score["spellF1"]), flush=True)
    means = {name: float(np.mean([result[name]["spellF1"] for result in report["conditions"].values()]))
             for name in ("possession_v4",)}
    report["mean_spellF1"] = means
    (args.output_dir / "external_benchmark.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
