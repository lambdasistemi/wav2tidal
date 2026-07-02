"""Config -> renderer params mapping (dirt.py, design-change-001)."""

from __future__ import annotations

from wav2tidal.core.pattern.dirt import nrt_params, rt_params
from wav2tidal.core.pattern.params import midicps


def test_rt_params_pass_everything_through():
    controls = {"note": 7.0, "cutoff": 1200.0, "vowel": "a", "room": 0.4}
    assert rt_params(controls) == controls
    assert rt_params(controls, n=3)["n"] == 3


def test_rt_params_do_not_clobber_explicit_n():
    assert rt_params({"n": 5.0}, n=3)["n"] == 5.0


def test_nrt_params_compute_freq_from_note():
    p = nrt_params("supersaw", {"note": 7.0}, sustain=2.0)
    assert abs(p["freq"] - midicps(67.0)) < 1e-9
    assert p["sustain"] == 2.0


def test_nrt_params_keep_only_synth_args():
    controls = {
        "note": 0.0,
        "lfo": 2.0,  # supersaw arg -> kept
        "cutoff": 1200.0,  # dirt_lpf module -> dropped (no module chain in NRT)
        "vowel": "a",  # dirt_vowel module -> dropped
        "room": 0.4,  # global orbit FX -> dropped
        "pan": 0.25,  # core synth arg -> kept
    }
    p = nrt_params("supersaw", controls, sustain=1.0)
    assert set(p) == {"sustain", "freq", "lfo", "pan"}


def test_nrt_params_forward_drum_n():
    p = nrt_params("superkick", {}, sustain=1.0, n=7)
    assert p["n"] == 7.0
