"""Tracking features and sequence utilities for the final possession model.

The benchmark reference is derived from OTHER matches' published events, not
the competition labels. Feature extraction never accepts an event table.
"""

from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402
from starter.tracking_io import team_of_each_track  # noqa: E402
from starter.tracking_io import read_tracking  # noqa: E402

LABELS = np.array(["H", "A", "D", "X"])
SAMPLE_REVISION = "e706dd506b360d69d9d123d5b8026e7294b13996"
SAMPLE_ROOT = ROOT / "data/external/metrica-samples"


@dataclass
class Tracking:
    ball: np.ndarray
    distances: np.ndarray
    movement: np.ndarray  # player speed median, 75th percentile, fraction above 1.5 m/s
    blocks: list
    fps: float = 25.0
    pitch: tuple = (105.0, 68.0)


def active_mask(n, blocks):
    mask = np.zeros(n, dtype=bool)
    previous_end = -1
    for _, first, last in blocks:
        if not 0 <= first <= last < n or first <= previous_end:
            raise ValueError("blocks must be ordered, disjoint and inside the timeline")
        mask[first:last + 1] = True
        previous_end = last
    return mask


def player_movement(speeds):
    speeds = np.where((speeds >= 0) & (speeds <= 15), speeds, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(speeds, axis=1)
        q75 = np.nanpercentile(speeds, 75, axis=1)
    count = np.isfinite(speeds).sum(axis=1)
    moving = np.divide(np.sum(speeds > 1.5, axis=1), count,
                       out=np.full(len(count), np.nan), where=count > 0)
    return np.column_stack((median, q75, moving))


def competition_tracking(data_dir=None):
    meta, raw = require(["atd_meta", "atd_raw"], data_dir=data_dir)
    teams = team_of_each_track(meta)
    game = GAMES[GAME]
    ball, distances = read_tracking(raw, teams, game["n_video_frames"], game["pitch"])
    speeds = np.full((len(ball), len(teams)), np.nan, dtype=np.float32)
    with open(raw) as handle:
        for frame, line in enumerate(handle):
            players = np.fromstring(line.split(":")[1].replace(";", ","), sep=",")
            if players.size != len(teams) * 3:
                raise ValueError("unexpected player layout at record {}".format(frame + 1))
            speeds[frame] = players.reshape(-1, 3)[:, 2]
    return Tracking(ball, distances, player_movement(speeds), half_video_blocks(GAME),
                    game["fps"], game["pitch"])


def sample_tracking(game, root=SAMPLE_ROOT):
    if game not in (1, 2):
        raise ValueError("only the two pinned OTHER-match sample games are supported")
    stem = "Sample_Game_{}".format(game)
    folder = Path(root) / "data" / stem
    sides = [pd.read_csv(folder / (stem + "_RawTrackingData_{}_Team.csv".format(side)),
                         skiprows=3, header=None).to_numpy(dtype=float) for side in ("Home", "Away")]
    home, away = sides
    if not np.array_equal(home[:, :3], away[:, :3]):
        raise ValueError("sample timelines differ between teams")
    if not np.array_equal(home[:, 1], np.arange(1, len(home) + 1)):
        raise ValueError("sample frame IDs must be 1..n")
    # Both files repeat the same ball columns; use home, rejecting disagreement.
    if not np.allclose(home[:, -2:], away[:, -2:], equal_nan=True):
        raise ValueError("sample ball columns disagree")
    scale = np.array([105.0, 68.0])
    ball = home[:, -2:] * scale
    positions = [side[:, 3:-2].reshape(len(side), -1, 2) * scale for side in sides]
    blocks = []
    for period in sorted(set(home[:, 0].astype(int))):
        index = np.flatnonzero(home[:, 0] == period)
        if not np.all(np.diff(index) == 1):
            raise ValueError("sample half is not contiguous")
        blocks.append((period, int(index[0]), int(index[-1])))
    distances, velocities = [], []
    for players in positions:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            distances.append(np.nanmin(np.linalg.norm(players - ball[:, None, :], axis=2), axis=1))
        speed = np.full(players.shape[:2], np.nan)
        # 0.4 s displacement is less sensitive to coordinate rounding than a
        # one-frame difference; never calculate across a half boundary.
        lag = 10
        for _, first, last in blocks:
            if last - first >= lag:
                speed[first + lag:last + 1] = np.linalg.norm(
                    players[first + lag:last + 1] - players[first:last + 1 - lag], axis=2) * 25 / lag
        velocities.append(speed)
    return Tracking(ball, np.column_stack(distances), player_movement(np.column_stack(velocities)), blocks)


def sample_reference(game, tracking, root=SAMPLE_ROOT):
    """Event-derived proxy possession reference, not manually corrected frame GT.

    Deliberate actions/recoveries establish a team; ball-out/goal/out shots and
    received fouls establish D. Labels carry to the next such marker within the
    same period. Challenges are deliberately excluded. Last uncovered frames are X.
    """
    stem = "Sample_Game_{}".format(game)
    events = pd.read_csv(Path(root) / "data" / stem / (stem + "_RawEventsData.csv"))
    labels = np.full(len(tracking.ball), 3, dtype=np.int8)
    for period, first, last in tracking.blocks:
        rows = events[events.Period == period]
        markers = []
        for row in rows.to_dict("records"):
            kind = row["Type"]
            subtype = str(row["Subtype"]).upper()
            start, end = int(row["Start Frame"]) - 1, int(row["End Frame"]) - 1
            team = {"Home": 0, "Away": 1}.get(row["Team"])
            if team is not None and kind in {"PASS", "SHOT", "BALL LOST", "BALL OUT", "RECOVERY", "SET PIECE"}:
                # Explicit priorities avoid a home/away bias when a loss and
                # recovery share a timestamp: recovery/restart wins the tie.
                priority = {"BALL LOST": 1, "BALL OUT": 1, "SHOT": 2,
                            "PASS": 3, "RECOVERY": 4, "SET PIECE": 5}[kind]
                markers.append((max(first, start), priority, team))
            if kind == "BALL OUT" or (kind == "SHOT" and ("GOAL" in subtype or "OUT" in subtype)):
                markers.append((max(first, start, end), 0, 2))
            if kind == "FAULT RECEIVED":
                markers.append((max(first, start), 0, 2))
        # At equal timestamps a restart/control action wins over a stop.
        markers.sort()
        final = min(last + 1, max(int(rows["End Frame"].max()), int(rows["Start Frame"].max())))
        for index, (start, _, label) in enumerate(markers):
            end = markers[index + 1][0] if index + 1 < len(markers) else final
            labels[max(first, start):min(last + 1, end)] = label
    return labels


def rolling_mean(values, radius):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    cumulative = np.r_[0.0, np.cumsum(np.where(valid, values, 0))]
    counts = np.r_[0, np.cumsum(valid)]
    index = np.arange(len(values))
    left, right = np.maximum(0, index - radius), np.minimum(len(values), index + radius + 1)
    total, count = cumulative[right] - cumulative[left], counts[right] - counts[left]
    return np.divide(total, count, out=np.full(len(values), np.nan), where=count > 0)


def neighbour_age(mask, fps):
    """Seconds since previous and until next observation, capped at 30 s."""
    n = len(mask)
    index = np.arange(n)
    previous = np.maximum.accumulate(np.where(mask, index, -int(30 * fps)))
    following = np.minimum.accumulate(np.where(mask, index, n + int(30 * fps))[::-1])[::-1]
    return np.minimum((index - previous) / fps, 30), np.minimum((following - index) / fps, 30)


def feature_matrix(tracking, dropout_seed=None):
    """Offline features from tracking ONLY; no labels or clock/game identity features.

    Synthetic contiguous dropouts are independent of class. They make training
    less dependent on the unusually clean external sample tracking.
    """
    n, fps = len(tracking.ball), tracking.fps
    if (not np.isfinite(fps) or fps <= 0 or tracking.ball.shape != (n, 2)
            or tracking.distances.shape != (n, 2) or tracking.movement.shape != (n, 3)):
        raise ValueError("tracking shapes or fps are invalid")
    active_mask(n, tracking.blocks)
    rng = np.random.default_rng(dropout_seed)
    outputs, names = [], None
    for _, first, last in tracking.blocks:
        ball = tracking.ball[first:last + 1].copy()
        distances = tracking.distances[first:last + 1].copy()
        size = len(ball)
        if dropout_seed is not None:
            # Alternate observed/missing blocks of varying duration, using only
            # a seeded process and never the external or competition labels.
            missing = np.zeros(size, dtype=bool)
            position = 0
            while position < size:
                position += int(rng.uniform(0.5, 4.0) * fps)
                end = min(size, position + max(1, int(rng.uniform(0.2, 4.0) * fps)))
                missing[position:end] = True
                position = end
            ball[missing] = np.nan
        visible = np.all(np.isfinite(ball), axis=1)
        distances[~visible] = np.nan
        distances[~np.isfinite(distances)] = np.nan
        distances = np.minimum(distances, 30)
        speed = np.full(size, np.nan)
        delta = np.linalg.norm(np.diff(ball, axis=0), axis=1) * fps
        speed[1:] = np.where(delta <= 45, delta, np.nan)
        # Reject a teleport's landing observation as evidence too.
        jumped = np.r_[False, np.isfinite(delta) & (delta > 45)]
        visible[jumped] = False
        distances[jumped] = np.nan
        ball[jumped] = np.nan
        boundary = np.min(np.column_stack((ball[:, 0], tracking.pitch[0] - ball[:, 0],
                                          ball[:, 1], tracking.pitch[1] - ball[:, 1])), axis=1)
        speed = np.minimum(speed, 45)
        features = {}
        features["ball_visible"] = visible.astype(float)
        features["edge_distance"] = np.clip(boundary, -10, 30)
        features["ball_speed"] = speed
        previous, following = neighbour_age(visible, fps)
        features["previous_ball_age"] = np.log1p(previous)
        features["next_ball_age"] = np.log1p(following)
        for team, side in enumerate(("home", "away")):
            features[side + "_distance"] = distances[:, team]
            other = distances[:, 1 - team]
            # A missing opposite side is not evidence of infinite advantage.
            control = visible & (distances[:, team] <= 2.5) & (other - distances[:, team] >= 0.75)
            previous, following = neighbour_age(control, fps)
            features[side + "_previous_control_age"] = np.log1p(previous)
            features[side + "_next_control_age"] = np.log1p(following)
            for span in (1.0, 3.0):
                radius = max(1, round(span * fps / 2))
                features[side + "_control_share_" + str(span)] = rolling_mean(control.astype(float), radius)
                features[side + "_distance_mean_" + str(span)] = rolling_mean(distances[:, team], radius)
        features["distance_advantage"] = np.tanh((distances[:, 1] - distances[:, 0]) / 3)
        for span in (0.5, 2.0, 6.0):
            radius = max(1, round(span * fps / 2))
            features["visibility_" + str(span)] = rolling_mean(visible.astype(float), radius)
            features["speed_mean_" + str(span)] = rolling_mean(speed, radius)
            stationary = np.where(np.isfinite(speed), (speed < 0.6).astype(float), np.nan)
            features["stationary_share_" + str(span)] = rolling_mean(stationary, radius)
        for index, name in enumerate(("player_speed_median", "player_speed_q75", "player_moving_fraction")):
            values = tracking.movement[first:last + 1, index]
            features[name] = values
            for span in (1.0, 3.0):
                features[name + "_mean_" + str(span)] = rolling_mean(values, max(1, round(span * fps / 2)))
        if names is None:
            names = list(features)
        outputs.append((first, last, np.column_stack(list(features.values()))))
    if names is None:
        raise ValueError("at least one tracking block is required")
    matrix = np.full((n, len(names)), np.nan, dtype=np.float32)
    for first, last, values in outputs:
        matrix[first:last + 1] = values
    return matrix, names


def mirrored_features(matrix, names):
    mirrored = matrix.copy()
    for index, name in enumerate(names):
        if name.startswith("home_"):
            other = names.index("away_" + name[5:])
            mirrored[:, index], mirrored[:, other] = matrix[:, other], matrix[:, index]
        elif name == "distance_advantage":
            mirrored[:, index] *= -1
    return mirrored


def emission_probabilities(model, matrix, names):
    # Averaging with a mirrored prediction enforces home/away symmetry even
    # when a single anonymised training match has different team styles.
    if not np.array_equal(model.classes_, [0, 1, 2]):
        raise ValueError("model must have H/A/D classes in order")
    forward = model.predict_proba(matrix)
    mirrored = model.predict_proba(mirrored_features(matrix, names))[:, [1, 0, 2]]
    probability = np.clip((forward + mirrored) / 2, 1e-6, 1)
    return probability / probability.sum(axis=1, keepdims=True)


def decode(probabilities, blocks, fps, switch_cost):
    """Offline three-state Viterbi decoder; transition cost is in log-loss seconds.

    Emission cost is integrated over time rather than multiplied by FPS. Each
    half is decoded separately. There is no forward fill, timer or label tuning.
    """
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(switch_cost) or switch_cost < 0:
        raise ValueError("fps must be positive and switch_cost finite/nonnegative")
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("probabilities must have shape (frames, 3)")
    if (not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-4)):
        raise ValueError("probabilities must be finite, nonnegative and sum to one")
    n = len(probabilities)
    active_mask(n, blocks)
    labels = np.full(n, 3, dtype=np.int8)
    transitions = np.full((3, 3), switch_cost)
    np.fill_diagonal(transitions, 0)
    for _, first, last in blocks:
        emissions = -np.log(np.clip(probabilities[first:last + 1], 1e-6, 1)) / fps
        back = np.zeros((len(emissions), 3), dtype=np.int8)
        score = emissions[0].copy()
        for frame in range(1, len(emissions)):
            paths = score[:, None] + transitions
            back[frame] = np.argmin(paths, axis=0)
            score = paths[back[frame], np.arange(3)] + emissions[frame]
            score -= score.min()  # numerical stability on long halves
        state = int(np.argmin(score))
        for frame in range(len(emissions) - 1, -1, -1):
            labels[first + frame] = state
            state = int(back[frame, state])
    return labels
