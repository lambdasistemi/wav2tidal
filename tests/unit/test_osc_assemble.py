"""OSC encoding and session assembly — pure tests (issue #51)."""

from __future__ import annotations

import numpy as np
import pytest

from wav2tidal.core.osc import message
from wav2tidal.core.render.assemble import assemble


def test_osc_message_layout():
    m = message("/w2t/load", "/tmp/x.scd")
    assert m.startswith(b"/w2t/load\x00")
    assert b",s\x00\x00" in m and b"/tmp/x.scd\x00" in m
    assert len(m) % 4 == 0


def test_osc_message_types():
    m = message("/a", 3, 0.5, "s")
    assert b",ifs" in m and len(m) % 4 == 0
    with pytest.raises(TypeError):
        message("/a", [1])


def test_assemble_places_and_normalizes():
    sr = 1000
    a = np.ones(500, dtype=np.float32) * 0.5  # mono -> stereo
    b = np.ones((500, 2), dtype=np.float32) * 0.5
    out = assemble([(0.0, a), (1.0, b)], total_seconds=2.0, sr=sr)
    assert out.shape == (2000, 2)
    assert abs(float(np.abs(out).max()) - 0.891) < 1e-3
    assert float(np.abs(out[600:900]).max()) == 0.0  # gap between clips
    # overlap sums, then everything normalizes together
    out2 = assemble([(0.0, a), (0.25, a)], 1.0, sr)
    assert float(np.abs(out2[300:400]).max()) > float(np.abs(out2[0:200]).max()) - 1e-6


def test_assemble_clips_at_canvas_end():
    sr = 1000
    out = assemble([(0.9, np.ones(500, dtype=np.float32))], 1.0, sr)
    assert out.shape == (1000, 2)
