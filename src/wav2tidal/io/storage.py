"""On-disk layout and the corpus manifest (T012).

Defines the local storage tree (all gitignored, FR-007) and the manifest
that makes ingestion idempotent and incremental (FR-006): a map of corpus
file -> (size, mtime, content hash). Re-ingest processes only files whose
hash changed or are new.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workspace:
    """Resolved output locations for one project root."""

    root: Path

    @property
    def banks(self) -> Path:
        return self.root / "banks"

    @property
    def profile(self) -> Path:
        return self.root / "profile"

    @property
    def manifest_path(self) -> Path:
        return self.profile / "corpus_manifest.json"

    def ensure(self) -> None:
        self.banks.mkdir(parents=True, exist_ok=True)
        self.profile.mkdir(parents=True, exist_ok=True)


def file_hash(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes (stable slice/track identity source)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def build_manifest(paths: list[Path]) -> dict[str, dict]:
    """Manifest entry per file: size, mtime, content hash."""
    manifest: dict[str, dict] = {}
    for p in sorted(paths):
        stat = p.stat()
        manifest[str(p)] = {
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "hash": file_hash(p),
        }
    return manifest


def load_manifest(path: str | Path) -> dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def save_manifest(path: str | Path, manifest: dict[str, dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def diff_manifest(old: dict[str, dict], new: dict[str, dict]) -> list[str]:
    """Files that are new or whose content hash changed (need processing)."""
    changed = []
    for path, entry in new.items():
        prev = old.get(path)
        if prev is None or prev.get("hash") != entry["hash"]:
            changed.append(path)
    return sorted(changed)
