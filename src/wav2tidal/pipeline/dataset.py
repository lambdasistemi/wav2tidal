"""Synthetic dataset generation (T033, FR-014 + design-change-001).

Generates seeded (style-descriptor-text -> config-text) pairs: sample a
valid config, render it, describe the audio, and pair the description
with the config text.

Two modes (``DatasetConfig.mode``):

- ``slices`` (v1, ``synth_dataset``): sample patterns over the ingested
  banks, pure numpy mixdown. Byte-deterministic from (config, seed)
  (SC-008); CI-safe.
- ``synth`` (v2, ``config_dataset``, issue #21): grammar-v2 configs over
  the Super* palette + banks, routed per config to the cheapest faithful
  renderer (numpy mix / headless NRT / booted-SuperDirt RT capture — see
  ``core.pattern.dirt.route``). Config *text* stays byte-deterministic
  from the seed; RT audio (and hence descriptors) is reproducible only
  within tolerance (SC-008 relaxation recorded in the artifact).

The descriptor text is the ByT5 model's input surface (T035); keep it
stable and bucketed so the mapping is learnable.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.config import DatasetConfig
from ..core.dsp.features import (
    centroid_motion,
    estimate_key,
    estimate_tempo,
    onset_rate,
)
from ..core.pattern.dirt import (
    MIX,
    NRT,
    RT,
    render_events,
    route,
    scene_plan,
    scene_route,
)
from ..core.pattern.generate import (
    Diversity,
    generate_config,
    generate_pattern,
    generate_scene,
)
from ..core.pattern.validate import (
    PatternBounds,
    SceneBounds,
    Sources,
    validate,
    validate_scene,
)
from ..core.render.mixdown import Banks, render
from ..core.render.schedule import schedule_events
from ..io.banks import load_banks
from ..io.storage import Workspace
from ..io.wav import read_wav, write_wav

_DENSITY_EDGES = (2.0, 6.0)  # onsets/sec -> lo / mid / hi
_BRIGHT_EDGES = (1500.0, 3000.0, 5000.0, 7000.0)  # spectral centroid Hz -> 1..5


def _bucket(value: float, edges) -> int:
    return sum(1 for e in edges if value >= e)


def _motion_label(ratio: float, wobble: float) -> str:
    if ratio >= 1.25:
        return "rising"
    if ratio <= 0.8:
        return "falling"
    return "wobbly" if wobble >= 0.15 else "steady"


def descriptor_text(audio: np.ndarray, sr: int, hop_length: int = 512) -> str:
    """Compact, bucketed description of a rendered clip — the model input.

    ``motion`` is the movement-aware field (issue #30): the direction/
    oscillation of the spectral-centroid track, so a filter sweep and a
    fixed timbre of equal average brightness get different descriptions.
    """
    import librosa

    bpm, _ = estimate_tempo(audio, sr, hop_length)
    key, _ = estimate_key(audio, sr, hop_length)
    density = onset_rate(audio, sr, hop_length)
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    density_label = ("lo", "mid", "hi")[_bucket(density, _DENSITY_EDGES)]
    brightness = _bucket(centroid, _BRIGHT_EDGES) + 1
    motion = _motion_label(*centroid_motion(audio, sr, hop_length))
    return (
        f"tempo={int(round(bpm))} density={density_label} "
        f"key={key} brightness={brightness}/5 motion={motion}"
    )


@dataclass
class DatasetResult:
    path: Path
    n_pairs: int


def dataset_id(cfg: DatasetConfig) -> str:
    return f"{cfg.mode}_n{cfg.size}_seed{cfg.seed}"


def synth_dataset(root: Path, cfg: DatasetConfig) -> DatasetResult:
    ws = Workspace(root)
    banks: Banks = load_banks(ws.banks, cfg.target_sr)
    inv = banks.inventory()
    if not inv:
        raise ValueError(f"no banks at {ws.banks} — run `wav2tidal ingest` first")

    rng = random.Random(cfg.seed)
    div = Diversity()
    bounds = PatternBounds(cfg.max_events_per_cycle, cfg.max_nesting_depth)
    total_seconds = cfg.n_cycles / cfg.cps

    out_dir = root / "datasets" / dataset_id(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "pairs.jsonl"

    n = 0
    with open(pairs_path, "w") as fh:
        while n < cfg.size:
            pattern = generate_pattern(rng, inv, div)
            if not validate(pattern, inv, bounds).valid:
                continue  # generator is valid by construction; belt and braces
            events = schedule_events(pattern, cfg.cps, cfg.n_cycles)
            audio = render(events, banks, total_seconds, cfg.target_sr)
            fh.write(
                json.dumps(
                    {
                        "input": descriptor_text(audio, cfg.target_sr, cfg.hop_length),
                        "output": pattern.to_text(),
                    }
                )
                + "\n"
            )
            n += 1

    (out_dir / "config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, sort_keys=True)
    )
    return DatasetResult(path=out_dir, n_pairs=n)


# -- v2 synth-path dataset (design-change-001, issue #21) --------------------

# SC-008 relaxation, embedded verbatim in every synth-mode artifact.
_REPRODUCIBILITY = {
    "config_text": "byte-deterministic from (config, seed)",
    "scenes": (
        "NRT scene audio byte-deterministic incl. trajectory automation; "
        "RT scenes within tolerance (booted SuperDirt)"
    ),
    MIX: "audio byte-deterministic",
    NRT: "audio byte-deterministic (scsynth NRT; seeded where defs carry RNG)",
    RT: (
        "audio reproducible within tolerance only (live SuperDirt capture; "
        "global FX use server RNG, scheduling is wall-clock) — expect "
        "descriptor buckets to match on re-render, not bytes"
    ),
}

# Renderer callables, injectable for pure tests:
#   rt_batch(jobs, banks_dir=...) -> list[Path]   (io.superdirt.rt_render_batch)
#   nrt_events(events, seconds, out) -> Path      (io.superdirt.nrt_render_events)
#   rt_scenes(jobs, banks_dir=...) -> list[Path]  (io.superdirt.rt_render_scene_batch)
#   nrt_scene(plan, out) -> Path                  (io.superdirt.nrt_render_scene)
RtBatch = Callable[..., list[Path]]
NrtEvents = Callable[..., Path]


def config_dataset(
    root: Path,
    cfg: DatasetConfig,
    *,
    sources: Sources | None = None,
    rt_batch: RtBatch | None = None,
    nrt_events: NrtEvents | None = None,
    rt_scenes: RtBatch | None = None,
    nrt_scene: NrtEvents | None = None,
) -> DatasetResult:
    """Generate (captured-audio descriptor -> grammar-v2 config) pairs.

    Configs are sampled first (pure, byte-deterministic from the seed),
    routed per config to the cheapest faithful renderer, rendered (RT jobs
    batched through one booted SuperDirt per ``rt_batch_size``), then
    described. Rendered audio is kept under ``audio/`` as the pairs'
    provenance. Pair order is generation order, so ``pairs.jsonl``'s
    ``output`` column is identical across re-runs; ``input`` is exact for
    mix/NRT rows and tolerance-reproducible for RT rows.
    """
    ws = Workspace(root)
    banks = (
        load_banks(ws.banks, cfg.target_sr)
        if ws.banks.is_dir()
        else Banks(sr=cfg.target_sr, data={})
    )
    inv = banks.inventory()
    sources = sources or Sources(banks=inv)

    rng = random.Random(cfg.seed)
    bounds = PatternBounds(cfg.max_events_per_cycle, cfg.max_nesting_depth)
    total_seconds = cfg.n_cycles / cfg.cps + cfg.tail_seconds

    scene_bounds = SceneBounds(pattern=bounds)
    can_scene = bool(sources.synths | sources.custom)
    items = []
    while len(items) < cfg.size:
        if can_scene and rng.random() < cfg.scene_ratio:
            scene = generate_scene(rng, sources)
            if not validate_scene(scene, sources, scene_bounds).valid:
                continue  # generator is valid by construction; belt and braces
            try:
                items.append(("scene", scene, scene_route(scene, sources)))
            except ValueError:
                continue  # unrenderable (e.g. vowel voice) — resample
        else:
            pattern = generate_config(rng, sources)
            if not validate(pattern, sources, bounds).valid:
                continue
            items.append(("line", pattern, route(pattern, sources)))

    out_dir = root / "datasets" / dataset_id(cfg)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    def _plan(scene):
        return scene_plan(scene, sources, total_seconds, cfg.cps, cfg.automation_tick)

    audio: dict[int, np.ndarray] = {}
    rt_jobs: list[tuple[int, Path]] = []
    rt_scene_jobs: list[tuple[int, Path]] = []
    for i, (kind, obj, mode) in enumerate(items):
        wav_path = audio_dir / f"{i:05d}.wav"
        if kind == "scene":
            if mode == NRT:
                fn = nrt_scene or _default_nrt_scene()
                fn(_plan(obj), wav_path)
            else:
                rt_scene_jobs.append((i, wav_path))
        elif mode == MIX:
            events = schedule_events(obj, cfg.cps, cfg.n_cycles)
            audio[i] = render(events, banks, total_seconds, cfg.target_sr)
            write_wav(wav_path, audio[i], cfg.target_sr)
        elif mode == NRT:
            fn = nrt_events or _default_nrt_events()
            fn(
                render_events(obj, sources, cfg.cps, cfg.n_cycles, NRT),
                total_seconds,
                wav_path,
            )
        else:
            rt_jobs.append((i, wav_path))

    banks_dir = ws.banks if inv else None
    fn = rt_batch or _default_rt_batch()
    for chunk_start in range(0, len(rt_jobs), cfg.rt_batch_size):
        chunk = rt_jobs[chunk_start : chunk_start + cfg.rt_batch_size]
        fn(
            [
                (
                    path,
                    total_seconds,
                    render_events(items[i][1], sources, cfg.cps, cfg.n_cycles, RT),
                )
                for i, path in chunk
            ],
            banks_dir=banks_dir,
        )
    fn = rt_scenes or _default_rt_scenes()
    for chunk_start in range(0, len(rt_scene_jobs), cfg.rt_batch_size):
        chunk = rt_scene_jobs[chunk_start : chunk_start + cfg.rt_batch_size]
        fn(
            [(path, _plan(items[i][1])) for i, path in chunk],
            banks_dir=banks_dir,
        )

    n = 0
    with open(out_dir / "pairs.jsonl", "w") as fh:
        for i, (kind, obj, mode) in enumerate(items):
            y = audio.get(i)
            if y is None:
                y = read_wav(audio_dir / f"{i:05d}.wav", cfg.target_sr).y
            fh.write(
                json.dumps(
                    {
                        "input": descriptor_text(y, cfg.target_sr, cfg.hop_length),
                        "output": obj.to_text(),
                        "renderer": mode,
                        "kind": kind,
                    }
                )
                + "\n"
            )
            n += 1

    (out_dir / "config.json").write_text(
        json.dumps(
            {
                **cfg.to_dict(),
                "sources": {
                    "banks": inv,
                    "synths": sorted(sources.synths),
                    "custom": sorted(sources.custom),
                },
                "reproducibility": _REPRODUCIBILITY,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return DatasetResult(path=out_dir, n_pairs=n)


def _default_rt_batch() -> RtBatch:
    from ..io.superdirt import rt_render_batch

    return rt_render_batch


def _default_nrt_events() -> NrtEvents:
    from ..io.superdirt import nrt_render_events

    return nrt_render_events


def _default_rt_scenes() -> RtBatch:
    from ..io.superdirt import rt_render_scene_batch

    return rt_render_scene_batch


def _default_nrt_scene() -> NrtEvents:
    from ..io.superdirt import nrt_render_scene

    return nrt_render_scene
