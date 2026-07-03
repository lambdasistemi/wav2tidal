"""Live session scripts — pure tests (issue #51)."""

from __future__ import annotations

from wav2tidal.core.pattern.dirt import scene_plan
from wav2tidal.core.pattern.model import Pattern, Scene, Trajectory, Voice
from wav2tidal.core.pattern.validate import Sources
from wav2tidal.io.superdirt import build_live_boot_script, build_live_swap_chunk

SOURCES = Sources(banks={"bd": 4})


def _plan(layer=None):
    scene = Scene(
        voices=(
            Voice(
                "supersaw",
                controls={"note": -12.0, "room": 0.4},
                mods=(Trajectory("cutoff", "ramp", (300.0, 3000.0)),),
            ),
        ),
        layer=layer,
    )
    return scene_plan(scene, SOURCES, 4.0, 0.5, 0.5)


def test_live_boot_script_structure():
    s = build_live_boot_script(port=57360, server_port=57160, banks_dir="/b")
    assert 'NetAddr("127.0.0.1", 57160)' in s  # dedicated server
    assert "~dirt.start(57360, [0]);" in s
    assert "CheckBadValues" in s  # sanitized monitor
    assert "OSCdef(\\w2t_load" in s and "'/w2t/load'" in s
    assert "W2T_LIVE_READY" in s
    assert '~dirt.loadSoundFiles("/b/*");' in s


def test_live_swap_chunk_swaps_and_loops():
    s = build_live_swap_chunk(_plan(layer=Pattern("bd bd:2", {})), port=57360)
    # new groups first, old freed after (hard cut with overlap, not a gap)
    assert s.index("~w2t_scene = Group.tail") < s.index("oldScene.free")
    assert s.index("nodes[\\v0] = Synth.tail(~w2t_scene") < s.index("oldScene.free")
    assert 'Synth.tail(~w2t_gfx, "dirt_reverb"' in s  # scene-owned reverb
    assert "nodes[\\v0_lpf].set(\\cutoff," in s  # trajectory ticks
    assert '"/dirt/play"' in s  # the layer
    assert "loop {" in s  # timeline repeats until the next swap
    assert "oldRoutine.stop;" in s and "oldBuses.do(_.free);" in s


def test_live_swap_chunk_without_globals_or_layer():
    s = build_live_swap_chunk(_plan())
    assert "dirt_delay" not in s
    assert "Group.tail(~orbit.group)" in s
