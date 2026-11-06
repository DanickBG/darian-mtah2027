"""ATD parsing, team metadata, hashing and submission output for the final model."""

import hashlib
import os
import xml.etree.ElementTree as ET
import numpy as np

HOME, AWAY = 0, 1

def team_of_each_track(meta_path):
    """Array over track slots: 0 if the slot belongs to the home team, 1 if away.

    The ATD metadata lists both teams and assigns every track slot to one of them. The first team
    in the file is the home side -- the FIFA/EPTS convention, and the one kloppy follows when it
    reports metadata.teams. For this match that is Barcelona, which DATA.md states independently.

    Note what these slots are NOT: there are 46 of them for 22 players on the pitch, they are
    named "Track 1".."Track 46", and their ShirtNumber is just the slot index. A slot is a
    tracklet, not a person, and it does not follow one player through the match.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))

    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([HOME if team_id == home_id else AWAY for team_id in slots], dtype=np.int8)

def write_submission(path, labels):
    """A frame,label CSV -- the whole submission format, in four lines."""
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label)
                          for frame, label in enumerate(labels))
    return path

def read_tracking(raw_path, teams, n_frames, pitch):
    """Stream raw ATD, checking every frame ID and slot count rather than silently shifting rows.

    No player identity continuity is assumed. Positions are normalised 0..1 in
    the file and converted to metres before comparing distances.
    """
    ball_xy = np.full((n_frames, 2), np.nan)
    distances = np.full((n_frames, 2), np.inf)
    scale = np.asarray(pitch, dtype=float)
    count = 0
    with open(raw_path) as handle:
        for index, line in enumerate(handle):
            try:
                frame, players, ball = line.strip().split(":")
                if index >= n_frames or int(frame) != index + 1:
                    raise ValueError("expected frame_id {}".format(index + 1))
                groups = players.split(";")
                if len(groups) != len(teams):
                    raise ValueError("track slot count differs from metadata")
                xy = np.array([float(value) for value in ball.split(",")[:2]])
                if xy.shape != (2,):
                    raise ValueError("ball must contain x,y")
                if np.all(np.isfinite(xy)):
                    ball_xy[index] = xy * scale
                    for slot, group in enumerate(groups):
                        player = np.array([float(value) for value in group.split(",")[:2]])
                        if player.shape != (2,):
                            raise ValueError("player must contain x,y")
                        if np.all(np.isfinite(player)):
                            distance = float(np.linalg.norm(player * scale - ball_xy[index]))
                            team = teams[slot]
                            distances[index, team] = min(distances[index, team], distance)
                count += 1
            except (ValueError, IndexError) as error:
                # Do not include the restricted tracking record in the error output.
                raise ValueError("invalid ATD record at line {}: {}".format(index + 1, error)) from error
    if count != n_frames:
        raise ValueError("expected {} ATD records, found {}".format(n_frames, count))
    return ball_xy, distances

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
