"""Pure test of the NRT sclang script builder (CI-safe, no SuperCollider)."""

from __future__ import annotations

from wav2tidal.io.superdirt import build_nrt_script


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
