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


# -- multi-event builders (issue #21) ----------------------------------------


def _batch():
    from wav2tidal.io.superdirt import build_rt_batch_script

    return build_rt_batch_script(
        jobs=[
            (
                "/tmp/a.wav",
                3.0,
                [
                    (0.0, "supersaw", {"note": 7, "room": 0.4}),
                    (1.0, "supersaw", {"note": 0}),
                ],
            ),
            ("/tmp/b.wav", 2.0, [(0.5, "bd", {"gain": 1.0})]),
        ],
        banks_dir="/ws/banks",
    )


def test_rt_batch_boots_once_and_records_per_job():
    s = _batch()
    assert s.count("SuperDirt(2, s)") == 1  # ONE boot for the whole batch
    assert 's.record("/tmp/a.wav", numChannels: 2);' in s
    assert 's.record("/tmp/b.wav", numChannels: 2);' in s
    assert s.count("s.stopRecording;") == 2
    assert "WAV2TIDAL_JOB_0_DONE" in s and "WAV2TIDAL_JOB_1_DONE" in s
    assert "WAV2TIDAL_RT_OK" in s


def test_rt_batch_schedules_event_waits():
    s = _batch()
    # job 0: event at 0, event at 1 -> 1.wait between, then 2.wait to 3.0s
    i0 = s.index("\\note, 7")
    i1 = s.index("\\note, 0")
    assert i0 < s.index("1.wait;", i0) < i1
    assert '~dirt.loadSoundFiles("/ws/banks/*");' in s


def test_rt_batch_realizes_delay_via_job_synth():
    # the orbit-owned dirt_delay never sounds after an event-driven resume
    # (verified on the box); delay params become a fresh per-job synth with
    # creation args and are stripped from the /dirt/play messages
    from wav2tidal.io.superdirt import build_rt_batch_script

    s = build_rt_batch_script(
        jobs=[
            (
                "/tmp/a.wav",
                3.0,
                [
                    (
                        0.0,
                        "supersaw",
                        {"note": 0, "delaytime": 0.25, "delayfeedback": 0.6},
                    )
                ],
            )
        ]
    )
    assert "\\delaytime, 0.25" in s and "\\delayfeedback, 0.6" in s
    assert s.index('Synth("dirt_delay"') < s.index('"/dirt/play"')
    assert "~w2t_delay.free;" in s
    # stripped from the event itself
    play = s[s.index('"/dirt/play"') : s.index("\\orbit, 0")]
    assert "delaytime" not in play and "delayfeedback" not in play


def test_rt_batch_without_delay_spawns_no_delay_synth():
    s = _batch()  # jobs use room, not delay
    assert "dirt_delay" not in s


def test_nrt_events_script_rows():
    from wav2tidal.io.superdirt import build_nrt_events_script

    s = build_nrt_events_script(
        events=[
            (1.0, "superkick", {"n": 3, "sustain": 0.5}),
            (0.0, "supersaw", {"freq": 220, "sustain": 1.0}),
        ],
        seconds=3.0,
        out_wav="/tmp/out.wav",
        osc_path="/tmp/score.osc",
        synthdef_files=["/quark/library/default-synths-extra.scd"],
    )
    # one d_recv per distinct synth (+ the RandSeed def), s_new rows in time
    # order with unique node ids, seed synth at score time 0
    assert s.count("d_recv") == 3
    assert "RandSeed.ir(1, seed)" in s and "\\w2t_seed, 999, 0, 0, \\seed, 1917" in s
    assert s.index("SynthDescLib.global[\\superkick]") < s.index("\\s_new, \\supersaw")
    assert s.index("[0, [\\s_new, \\supersaw, 1000") < s.index(
        "[1, [\\s_new, \\superkick, 1001"
    )
    assert "duration: 3" in s and "WAV2TIDAL_NRT_OK" in s
