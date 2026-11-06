"""V4 tracking perturbations and duration-constrained sequence decoding.

No function here reads the competition ground truth. Other-match event labels
are accessed separately, after tracking-only features have been constructed.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402
from starter.tracking_io import team_of_each_track  # noqa: E402
from starter.tracking_io import sha256  # noqa: E402
from starter.sequence_possession import (  # noqa: E402
    SAMPLE_ROOT, Tracking, active_mask, player_movement, sample_tracking,
)

THIRD_ROOT = ROOT / "data/external/metrica-game3"
V4_ROOT = ROOT / "data/derived/possession_v4"
PROFILES = ("clean", "gaps", "position", "combined")


@dataclass
class RichTracking(Tracking):
    players: tuple = ()


def distances_to_teams(ball, players):
    result = []
    for positions in players:
        distance = np.linalg.norm(positions - ball[:, None, :], axis=2)
        distance[~np.isfinite(distance)] = np.inf
        nearest = np.min(distance, axis=1)
        nearest[~np.isfinite(nearest)] = np.nan
        result.append(nearest)
    return np.column_stack(result)


def trajectory_movement(players, blocks, fps):
    speeds = []
    lag = max(1, round(0.4 * fps))
    for positions in players:
        speed = np.full(positions.shape[:2], np.nan, dtype=np.float32)
        for _, first, last in blocks:
            if last - first >= lag:
                speed[first + lag:last + 1] = np.linalg.norm(
                    positions[first + lag:last + 1] - positions[first:last + 1 - lag], axis=2) * fps / lag
        speeds.append(speed)
    return player_movement(np.column_stack(speeds))


def read_epts(meta, raw, n, blocks, fps, pitch, has_speed):
    teams = team_of_each_track(meta)
    slots = len(teams)
    scale = np.asarray(pitch)
    players = np.full((n, slots, 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan)
    speeds = np.full((n, slots), np.nan, dtype=np.float32) if has_speed else None
    width = 3 if has_speed else 2
    layouts = []
    if not has_speed:
        # The clean EPTS sample lists all squad members but writes only the
        # active players. Its format changes at substitutions, including order.
        xml = ET.parse(meta).getroot()
        squad = {player.get("id"): index for index, player in enumerate(xml.iter("Player"))}
        channels = {channel.get("id"): (channel.get("playerId"), channel.get("channelId"))
                    for channel in xml.iter("PlayerChannel")}
        for specification in xml.iter("DataFormatSpecification"):
            groups = specification.find("./SplitRegister[@separator=';']")
            if groups is None:
                raise ValueError("EPTS layout has no player groups")
            mapping = []
            for group in groups.findall("./SplitRegister"):
                references = [channels[ref.get("playerChannelId")] for ref in group.findall("PlayerChannelRef")]
                if (len(references) != 2 or references[0][0] != references[1][0]
                        or [ref[1] for ref in references] != ["x", "y"]):
                    raise ValueError("unsupported EPTS player-channel order")
                mapping.append(squad[references[0][0]])
            layouts.append((int(specification.get("startFrame")) - 1,
                            int(specification.get("endFrame")) - 1, np.array(mapping)))
        layouts.sort(key=lambda item: item[0])
    layout_index = 0
    count = 0
    with open(raw) as handle:
        for frame, line in enumerate(handle):
            parts = line.strip().split(":")
            if len(parts) != 3 or frame >= n or int(parts[0]) != frame + 1:
                raise ValueError("invalid tracking sequence at record {}".format(frame + 1))
            values = np.fromstring(parts[1].replace(";", ","), sep=",")
            mapping = np.arange(slots)
            if layouts:
                while layout_index + 1 < len(layouts) and frame > layouts[layout_index][1]:
                    layout_index += 1
                start, end, mapping = layouts[layout_index]
                if not start <= frame <= end:
                    raise ValueError("frame is not covered by an EPTS layout")
            if values.size != len(mapping) * width:
                raise ValueError("unexpected player layout at record {}".format(frame + 1))
            values = values.reshape(len(mapping), width)
            players[frame, mapping] = values[:, :2] * scale
            if has_speed:
                speeds[frame] = values[:, 2]
            observed_ball = np.fromstring(parts[2], sep=",")
            if len(observed_ball) < 2:
                raise ValueError("invalid ball layout at record {}".format(frame + 1))
            ball[frame] = observed_ball[:2] * scale
            count += 1
    if count != n:
        raise ValueError("tracking record count differs from metadata")
    grouped = tuple(players[:, teams == team] for team in (0, 1))
    movement = player_movement(speeds) if has_speed else trajectory_movement(grouped, blocks, fps)
    return RichTracking(ball, distances_to_teams(ball, grouped), movement, blocks, fps, pitch, grouped)


def cached_tracking(key, builder, cache):
    """Cache only numeric arrays, verified by source/input hashes; never load pickle."""
    cache = Path(cache)
    encoded = json.dumps(key, sort_keys=True)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            if str(saved["key"].item()) == encoded:
                return RichTracking(saved["ball"].copy(), saved["distances"].copy(), saved["movement"].copy(),
                                    [tuple(row) for row in saved["blocks"].tolist()], float(saved["fps"]),
                                    tuple(saved["pitch"].tolist()), (saved["home"].copy(), saved["away"].copy()))
    tracking = builder()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, key=encoded, ball=tracking.ball, distances=tracking.distances,
                        movement=tracking.movement, blocks=np.array(tracking.blocks), fps=tracking.fps,
                        pitch=np.array(tracking.pitch), home=tracking.players[0], away=tracking.players[1])
    return tracking


def tracking_sources(paths):
    return {
        "inputs": {str(path): sha256(path) for path in paths},
        "sources": {str(path.relative_to(ROOT)): sha256(path) for path in (
            Path(__file__), ROOT / "starter/sequence_possession.py", ROOT / "starter/tracking_io.py",
            ROOT / "starter/tracking_io.py", ROOT / "games.py")},
    }


def rich_sample(game, samples=SAMPLE_ROOT, third=THIRD_ROOT, cache_dir=V4_ROOT / "cache"):
    if game in (1, 2):
        stem = "Sample_Game_{}".format(game)
        paths = [Path(samples) / "data" / stem / (stem + "_RawTrackingData_{}_Team.csv".format(side))
                 for side in ("Home", "Away")]
        def build():
            tracking = sample_tracking(game, samples)
            players = tuple((pd.read_csv(path, skiprows=3, header=None).to_numpy(dtype=float)[:, 3:-2]
                             .reshape(len(tracking.ball), -1, 2) * np.asarray(tracking.pitch)).astype(np.float32)
                            for path in paths)
            return RichTracking(tracking.ball, tracking.distances, tracking.movement, tracking.blocks,
                                tracking.fps, tracking.pitch, players)
    elif game == 3:
        meta, raw = Path(third) / "Sample_Game_3_metadata.xml", Path(third) / "Sample_Game_3_tracking.txt"
        paths = [meta, raw]
        def build():
            xml = ET.parse(meta).getroot()
            parameters = {p.findtext("Name"): p.findtext("Value") for p in xml.findall(
                ".//ProviderGlobalParameters/ProviderParameter")}
            blocks = [(half, int(parameters[name + "_start"]) - 1, int(parameters[name + "_end"]) - 1)
                      for half, name in ((1, "first_half"), (2, "second_half"))]
            pitch = (float(xml.findtext(".//FieldSize/Width")), float(xml.findtext(".//FieldSize/Height")))
            return read_epts(meta, raw, blocks[-1][2] + 1, blocks, float(xml.findtext(".//FrameRate")), pitch, False)
    else:
        raise ValueError("unknown external sample")
    return cached_tracking(tracking_sources(paths), build, Path(cache_dir) / "game_{}.npz".format(game))


def rich_competition(data_dir=None, cache_dir=V4_ROOT / "cache"):
    meta, raw = [Path(path) for path in require(["atd_meta", "atd_raw"], data_dir=data_dir)]
    game = GAMES[GAME]
    return cached_tracking(tracking_sources([meta, raw]), lambda: read_epts(
        meta, raw, game["n_video_frames"], half_video_blocks(GAME), game["fps"], game["pitch"], True),
        Path(cache_dir) / "competition.npz")


def perturb(tracking, seed, profile):
    """Class-independent tracking errors; label arrays are never accepted.

    Position errors are correlated through time. Recompute distances from full
    geometry rather than injecting an arbitrary bias into the nearest team.
    """
    if profile not in PROFILES:
        raise ValueError("unknown perturbation profile")
    if profile == "clean":
        return Tracking(tracking.ball.copy(), tracking.distances.copy(), tracking.movement.copy(),
                        list(tracking.blocks), tracking.fps, tracking.pitch)
    if len(tracking.players) != 2:
        raise ValueError("full player positions are required to perturb tracking")
    rng = np.random.default_rng(seed)
    ball = tracking.ball.copy()
    players = tracking.players
    if profile in ("position", "combined"):
        sigma = 0.75 if profile == "position" else 1.25
        for _, first, last in tracking.blocks:
            noise = rng.normal(0, sigma, (last - first + 1, 2))
            ball[first:last + 1] += lfilter([math.sqrt(1 - 0.8 ** 2)], [1, -0.8], noise, axis=0)
            # Occasional brief false ball tracks, at random positions/times.
            starts = np.flatnonzero(rng.random(last - first + 1) < 0.0005) + first
            for start in starts:
                end = min(last + 1, start + int(rng.uniform(0.2, 0.6) * tracking.fps))
                ball[start:end] += rng.normal(0, 8, 2)
        perturbed = []
        for positions in players:
            values = positions + rng.normal(0, 0.2, positions.shape).astype(np.float32)
            values[rng.random(positions.shape[:2]) < 0.08] = np.nan
            perturbed.append(values)
        players = tuple(perturbed)
    if profile in ("gaps", "combined"):
        for _, first, last in tracking.blocks:
            position = first
            while position <= last:
                position += int(rng.uniform(0.5, 4.0) * tracking.fps)
                end = min(last + 1, position + max(1, int(rng.uniform(0.2, 4.0) * tracking.fps)))
                ball[position:end] = np.nan
                position = end
    return Tracking(ball, distances_to_teams(ball, players), tracking.movement.copy(),
                    list(tracking.blocks), tracking.fps, tracking.pitch)


def decode_duration(probabilities, blocks, fps, switch_cost, minimum_s=0.0, step=5):
    """Exact minimum-duration segment DP on fixed-size emission groups.

    No maximum duration and no target possession count. Original frame losses
    are summed, not discarded, and half boundaries are never crossed.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    if (probabilities.ndim != 2 or probabilities.shape[1] != 3
            or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-4)):
        raise ValueError("invalid state probabilities")
    if (not np.isfinite(fps) or fps <= 0 or not np.isfinite(switch_cost) or switch_cost < 0
            or not np.isfinite(minimum_s) or minimum_s < 0 or isinstance(step, bool)
            or not isinstance(step, int) or step <= 0):
        raise ValueError("invalid decoding settings")
    blocks = list(blocks)
    active_mask(len(probabilities), blocks)
    output = np.full(len(probabilities), 3, dtype=np.int8)
    transitions = np.full((3, 3), switch_cost, dtype=float)
    np.fill_diagonal(transitions, np.inf)
    for _, first, last in blocks:
        loss = -np.log(np.clip(probabilities[first:last + 1], 1e-6, 1)) / fps
        group_starts = np.arange(0, len(loss), step)
        grouped = np.add.reduceat(loss, group_starts, axis=0)
        n = len(grouped)
        cumulative = np.vstack((np.zeros((1, 3)), np.cumsum(grouped, axis=0)))
        boundaries = np.r_[group_starts, len(loss)]
        required_frames = min(len(loss), max(1, math.ceil(minimum_s * fps)))
        score = np.full((n + 1, 3), np.inf)
        starts = np.zeros((n + 1, 3), dtype=np.int32)
        previous = np.full((n + 1, 3), -1, dtype=np.int8)
        best = np.full(3, np.inf)
        best_start = np.zeros(3, dtype=np.int32)
        best_previous = np.full(3, -1, dtype=np.int8)
        added = -1
        for end in range(1, n + 1):
            # Account for a short final group in actual frames, not just group
            # counts, so the configured minimum is respected at the half end.
            eligible = np.searchsorted(boundaries, boundaries[end] - required_frames, side="right") - 1
            for begin in range(added + 1, int(eligible) + 1):
                if begin == 0:
                    incoming, predecessor = np.zeros(3), np.full(3, -1, dtype=np.int8)
                else:
                    candidates = score[begin, :, None] + transitions
                    predecessor = np.argmin(candidates, axis=0).astype(np.int8)
                    incoming = candidates[predecessor, np.arange(3)]
                value = incoming - cumulative[begin]
                better = value < best
                best[better], best_start[better], best_previous[better] = value[better], begin, predecessor[better]
            added = int(eligible)
            score[end] = cumulative[end] + best
            starts[end], previous[end] = best_start, best_previous
        state, end = int(np.argmin(score[n])), n
        while end > 0:
            begin = int(starts[end, state])
            left = first + int(group_starts[begin])
            right = first + (int(group_starts[end]) if end < n else len(loss))
            output[left:right] = state
            state, end = int(previous[end, state]), begin
    return output


def third_reference(tracking, root=THIRD_ROOT):
    """Public OTHER-match event proxy. Call only for the final reserved evaluation."""
    meta = ET.parse(Path(root) / "Sample_Game_3_metadata.xml").getroot()
    lookup = {team.get("id"): index for index, team in enumerate(meta.iter("Team"))}
    events = json.loads((Path(root) / "Sample_Game_3_events.json").read_text())["data"]
    labels = np.full(len(tracking.ball), 3, dtype=np.int8)
    priorities = {"BALL LOST": 1, "BALL OUT": 1, "SHOT": 2, "PASS": 3,
                  "CARRY": 3, "RECOVERY": 4, "SET PIECE": 5}
    for period, first, last in tracking.blocks:
        rows = [event for event in events if event["period"] == period]
        markers = []
        for event in rows:
            kind = event["type"]["name"]
            subtype = event.get("subtypes")
            if isinstance(subtype, dict):
                subtype = subtype.get("name", "")
            elif isinstance(subtype, list):
                subtype = "-".join(value.get("name", "") for value in subtype)
            subtype = str(subtype).upper()
            start = int(event["start"]["frame"]) - 1
            end = int(event["end"]["frame"]) - 1
            team = lookup.get((event.get("team") or {}).get("id"))
            if kind in priorities and team is not None:
                markers.append((max(first, start), priorities[kind], team))
            if kind == "BALL OUT" or (kind == "SHOT" and ("GOAL" in subtype or "OUT" in subtype)):
                markers.append((max(first, start, end), 0, 2))
            if kind == "FAULT RECEIVED":
                markers.append((max(first, start), 0, 2))
        markers.sort()
        final = min(last + 1, max(max(row["start"]["frame"], row["end"]["frame"]) for row in rows))
        for index, (start, _, label) in enumerate(markers):
            end = markers[index + 1][0] if index + 1 < len(markers) else final
            labels[max(first, start):min(last + 1, end)] = label
    return labels
