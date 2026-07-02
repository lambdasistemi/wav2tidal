"""Pure test of the sclang script builders (CI-safe, no SuperCollider)."""

from __future__ import annotations

from wav2tidal.io.superdirt import build_nrt_script, build_rt_script


def _script():
    return build_nrt_script(
        synth="supersaw",
        params={"freq": 220, "sustain": 1.0, "pan": 0.5},
        seconds=1.4,
        out_wav="/tmp/out.wav",
        osc_path="/tmp/score.osc",
        synthdef_files=["/quark/library/default-synths-extra.scd"],
        sr=44100,
    )


def test_script_contains_core_elements():
    s = _script()
    assert '"/quark/library/default-synths-extra.scd".load;' in s
    assert "SynthDescLib.global[\\supersaw].def.asBytes" in s
    assert "Score.recordNRT" in s
    assert '"/tmp/out.wav"' in s
    assert "WAV2TIDAL_NRT_OK" in s


def test_params_are_emitted_sorted():
    s = _script()
    # sorted keys: freq, pan, sustain
    assert s.index("\\freq") < s.index("\\pan") < s.index("\\sustain")


def test_duration_wired():
    s = _script()
    assert "duration: 1.4" in s
    assert "44100" in s


def _rt():
    return build_rt_script(
        synth="supersaw",
        params={"cutoff": 500, "room": 0.7, "note": 0},
        seconds=3.0,
        out_wav="/tmp/rt.wav",
    )


def test_rt_script_boots_superdirt_and_plays_dirt():
    s = _rt()
    assert "SuperDirt(2, s)" in s
    assert "~dirt.start(57120, [0])" in s
    assert 's.record("/tmp/rt.wav"' in s
    assert '"/dirt/play"' in s
    assert '\\s, "supersaw"' in s
    assert "WAV2TIDAL_RT_OK" in s


def test_rt_script_emits_fx_params():
    s = _rt()
    assert "\\cutoff, 500" in s
    assert "\\room, 0.7" in s
    assert "\\orbit, 0" in s
