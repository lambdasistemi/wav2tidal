"""Minimal OSC 1.0 message encoding (US3-2, issue #51).

Just enough to command the resident sclang session (string/int/float
arguments); no bundles, no parsing. Pure — no sockets here.
"""

from __future__ import annotations

import struct


def _pad(b: bytes) -> bytes:
    return b + b"\x00" * (4 - len(b) % 4 if len(b) % 4 else 0)


def _osc_string(s: str) -> bytes:
    return _pad(s.encode("utf-8") + b"\x00")


def message(address: str, *args: str | int | float) -> bytes:
    """Encode one OSC message."""
    tags = ","
    payload = b""
    for a in args:
        if isinstance(a, bool):
            raise TypeError("OSC bools not supported")
        if isinstance(a, int):
            tags += "i"
            payload += struct.pack(">i", a)
        elif isinstance(a, float):
            tags += "f"
            payload += struct.pack(">f", a)
        elif isinstance(a, str):
            tags += "s"
            payload += _osc_string(a)
        else:
            raise TypeError(f"unsupported OSC argument type: {type(a)!r}")
    return _osc_string(address) + _osc_string(tags) + payload
