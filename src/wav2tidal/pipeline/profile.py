"""Style-profile persistence and nearest-neighbour query (T023).

The profile stores per-track and per-slice descriptors plus a track-level
nearest-neighbour index. Metadata is JSON; descriptor vectors live in a
companion ``.npz`` (numpy). Query embeds/derives the query descriptor and
returns the k nearest tracks by cosine similarity (SC-002).
"""

from __future__ import annotations

import json

import numpy as np

from ..core.descriptor.types import ProfileIndex, StyleDescriptor
from ..io.storage import Workspace

_INDEX = "track_index.npz"
_TRACKS = "tracks.json"
_SLICES = "slices.json"


def save_profile(
    ws: Workspace,
    tracks: list[dict],
    slices: list[dict],
    index: ProfileIndex,
) -> None:
    ws.profile.mkdir(parents=True, exist_ok=True)
    (ws.profile / _TRACKS).write_text(json.dumps(tracks, indent=2, sort_keys=True))
    (ws.profile / _SLICES).write_text(json.dumps(slices, indent=2, sort_keys=True))
    np.savez(
        ws.profile / _INDEX,
        ids=np.array(index.ids, dtype=object),
        matrix=index.matrix,
        embedder_id=index.embedder_id,
        sr_used=index.sr_used,
    )


def load_index(ws: Workspace) -> ProfileIndex:
    data = np.load(ws.profile / _INDEX, allow_pickle=True)
    return ProfileIndex(
        ids=list(data["ids"]),
        matrix=data["matrix"],
        embedder_id=str(data["embedder_id"]),
        sr_used=int(data["sr_used"]),
    )


def load_tracks(ws: Workspace) -> list[dict]:
    path = ws.profile / _TRACKS
    return json.loads(path.read_text()) if path.exists() else []


def query_nearest(
    ws: Workspace, query: StyleDescriptor, k: int = 5
) -> list[tuple[str, float]]:
    return load_index(ws).nearest(query, k=k)


def resolve_query_id(ws: Workspace, query: str) -> str | None:
    """Map a query (track id or source path) to an index id, if present."""
    idx = load_index(ws)
    if query in idx.ids:
        return query
    for track in load_tracks(ws):
        if track.get("source_path") == query or track.get("source_stem") == query:
            return track["id"]
    return None
