"""The grammar-v2 parameter table (design-change-001, research R7).

The verified SuperDirt synth+FX control vocabulary, encoded: every
parameter the action space may set, with scope, kind, range, and per-synth
applicability. This module is the single semantic source of truth shared
by the generator (values in range by construction), the validator (range
checks), and the renderer mapping (event vs global scope, note->freq).
The *syntactic* source of truth is ``grammar/pattern_subset.lark`` v2; a
unit test asserts the two vocabularies agree.

Provenance: defaults and hard clips are read from the SuperDirt quark
sources (synths/core-synths.scd, synths/core-synths-global.scd,
synths/core-modules.scd, library/default-synths-extra.scd, quark v1.7.3);
sampling ranges are our chosen musical action space, documented per-param
in specs/001-corpus-to-live-pipeline/contracts/params-v2.md. Pure — no IO.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

VOWELS = ("a", "e", "i", "o", "u")

# Scopes: "event" params ride each /dirt/play message (the per-event FX
# chain and source-synth args); "global" params address the orbit FX
# (dirt_reverb / dirt_delay) and are only meaningful on the RT path.
EVENT = "event"
GLOBAL = "global"

# Kinds: "continuous" (uniform float), "log" (log-uniform float),
# "integer" (uniform int), "choice" (enumerated strings, e.g. vowel).
CONTINUOUS = "continuous"
LOG = "log"
INTEGER = "integer"
CHOICE = "choice"


@dataclass(frozen=True)
class ParamSpec:
    name: str
    lo: float = 0.0
    hi: float = 1.0
    kind: str = CONTINUOUS
    scope: str = EVENT
    choices: tuple[str, ...] = ()

    def sample(
        self, rng: random.Random, lo: float | None = None, hi: float | None = None
    ):
        lo = self.lo if lo is None else lo
        hi = self.hi if hi is None else hi
        if self.kind == CHOICE:
            return rng.choice(self.choices)
        if self.kind == INTEGER:
            return rng.randint(int(lo), int(hi))
        if self.kind == LOG and lo > 0:
            v = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        else:  # uniform; also for log params whose override reaches 0
            v = rng.uniform(lo, hi)
        v = float(f"{v:.3g}")
        if abs(v) < 1e-4 and lo <= 0:  # keep text decimal, never 2e-05
            v = 0.0
        return v

    def in_range(self, value: float | str) -> bool:
        if self.kind == CHOICE:
            return value in self.choices
        if isinstance(value, str):
            return False
        if self.kind == INTEGER and value != int(value):
            return False
        return self.lo <= value <= self.hi


def _p(name, lo=0.0, hi=1.0, kind=CONTINUOUS, scope=EVENT, choices=()):
    return ParamSpec(name, lo, hi, kind, scope, choices)


# -- Core event params (any source: sample bank, Super* synth, custom def).
# note: semitones relative to middle C (SuperDirt midinote = note + 60).
CORE = (
    _p("note", -24, 24),
    _p("n", 0, 24, kind=INTEGER),  # drum-synth pitch knob / sample index
    _p("gain", 0.5, 1.3),  # dirt_gate raises to the 4th power, caps at 2
    _p("pan", 0.0, 1.0),
    _p("speed", 0.25, 4.0, kind=LOG),
    _p("accelerate", -2.0, 2.0),
    _p("attack", 0.0, 0.5),  # triggers dirt_envelope (with release)
    _p("release", 0.02, 2.0),
)

# -- Per-event FX params (each activates its dirt_* effect synth when set;
# hard clips from core-synths.scd noted in contracts/params-v2.md).
EVENT_FX = (
    _p("cutoff", 60, 12000, kind=LOG),  # dirt_lpf; source clips 20..SR/2
    _p("resonance", 0.0, 0.8),  # dirt_lpf rq; also a direct synth arg
    _p("hcutoff", 60, 12000, kind=LOG),  # dirt_hpf
    _p("hresonance", 0.0, 0.8),
    _p("bandf", 60, 12000, kind=LOG),  # dirt_bpf
    _p("bandq", 1.0, 50.0, kind=LOG),  # source floors at 1
    _p("shape", 0.0, 0.95),  # source clamps < 1
    _p("crush", 1.0, 16.0),  # bit depth: round(0.5 ** (crush - 1))
    _p("coarse", 1, 32, kind=INTEGER),  # activates only when > 1
    _p("vowel", kind=CHOICE, choices=VOWELS),  # dirt_vowel formant filter
)

# -- Global (orbit) FX sends — RT path only; no NRT support (R7 tier 2).
GLOBAL_FX = (
    _p("room", 0.0, 1.0, scope=GLOBAL),  # reverb feed; activates dirt_reverb
    _p("size", 0.0, 1.0, scope=GLOBAL),  # reverb depth (linexp 0.01..0.98)
    _p("delaytime", 0.02, 1.0, scope=GLOBAL),  # source clips 0..4 s
    _p("delayfeedback", 0.0, 0.9, scope=GLOBAL),  # source clips 0..0.99
)

# -- Synth-specific params: default specs; per-synth (lo, hi) overrides in
# SYNTHS below. One event namespace: e.g. `resonance` reaching a supersaw
# is the same param that configures dirt_lpf.
SYNTH_PARAM_SPECS = (
    _p("voice", 0.0, 1.0),
    _p("semitone", 0.0, 24.0),
    _p("lfo", 0.0, 4.0),
    _p("pitch1", 0.25, 8.0, kind=LOG),
    _p("pitch2", 1.0, 4.0),
    _p("pitch3", 1.0, 6.0),
    _p("rate", 0.25, 4.0, kind=LOG),
    _p("decay", 0.0, 1.0),
    _p("detune", 0.0, 2.0),
    _p("slide", -4.0, 4.0),
    _p("velocity", 0.0, 1.2),
    _p("muffle", 0.0, 2.0),
    _p("stereo", 0.0, 1.0),
    _p("modamp", 0.0, 2.0),
    _p("modfreq", 2.0, 12.0),
    _p("vibrato", 0.0, 1.0),
    _p("vrate", 1.0, 10.0),
    _p("perc", 0.0, 1.2),
    _p("percf", 2, 3, kind=INTEGER),
    _p("lfofreq", 0.1, 10.0, kind=LOG),
    _p("lfodepth", 0.0, 0.5),
)

# The broad Super* palette (library/default-synths-extra.scd): synth name
# -> {param: (lo, hi) override or None for the default spec range}. Only
# params listed here (plus CORE and EVENT_FX) are applicable to a synth.
_MOOG = {
    "voice": None,
    "semitone": None,
    "resonance": None,
    "lfo": None,
    "pitch1": None,
    "rate": None,
    "decay": None,
}
SYNTHS: dict[str, dict[str, tuple[float, float] | None]] = {
    "supermandolin": {"detune": (0.0, 3.0)},
    "supergong": {"voice": (0.0, 4.0), "decay": (0.0, 2.0)},
    "superpiano": {
        "velocity": None,
        "detune": (0.0, 1.0),
        "muffle": None,
        "stereo": None,
    },
    "superhex": {"rate": None},
    "superkick": {"pitch1": (0.25, 4.0), "decay": (0.25, 2.0)},
    "super808": {"rate": (0.25, 4.0), "voice": (0.0, 2.0)},
    "superhat": {},
    "supersnare": {"decay": (0.25, 2.0)},
    "superclap": {"rate": (0.25, 4.0), "pitch1": (0.25, 4.0)},
    "supersiren": {},
    "supersquare": dict(_MOOG, voice=(0.05, 0.95)),  # width 0/1 is silent
    "supersaw": dict(_MOOG),
    "superpwm": dict(_MOOG),
    "supercomparator": {
        "voice": (0.0, 5.0),
        "resonance": None,
        "lfo": None,
        "pitch1": None,
        "rate": None,
        "decay": None,
    },
    "superchip": {
        "slide": None,
        "rate": (0.25, 4.0),
        "pitch2": None,
        "pitch3": None,
        "voice": None,
    },
    "supernoise": {
        "voice": None,
        "slide": (0.0, 4.0),
        "pitch1": None,
        "rate": (0.25, 4.0),
        "resonance": (0.0, 1.0),
    },
    "superfork": {},
    "superhammond": {
        "voice": (0.0, 9.0),
        "vibrato": None,
        "vrate": None,
        "perc": None,
        "percf": None,
        "decay": None,
    },
    "supervibe": {
        "decay": None,
        "velocity": (0.0, 1.5),
        "modamp": None,
        "modfreq": None,
        "detune": (0.0, 1.0),
    },
    "superhoover": {"slide": None, "decay": None},
    "superzow": {"slide": (0.25, 4.0), "detune": (0.0, 3.0), "decay": None},
    "superstatic": {},
    "supergrind": {"detune": (0.0, 10.0), "voice": None, "rate": (0.25, 4.0)},
    "superprimes": {"detune": (0.0, 1.0), "voice": (0.0, 2.0), "rate": (0.25, 4.0)},
    "superwavemechanics": {
        "detune": (0.0, 1.5),
        "voice": None,
        "resonance": (0.0, 1.0),
    },
    "supertron": {"voice": None, "detune": (0.0, 5.0)},
    "superreese": {"voice": (0.0, 2.0), "detune": None},
    # superfm's full 6-operator matrix (amp1..mod66) is out of the v2 action
    # space; we expose its presets + pitch LFO only.
    "superfm": {"voice": (0.0, 5.0), "lfofreq": None, "lfodepth": None},
    "soskick": {"pitch1": (0.0, 2000.0), "voice": (0.0, 4.0), "pitch2": (0.0, 1.0)},
    "soshats": {"pitch1": (50.0, 1000.0), "resonance": (0.0, 1.0)},
    "sostoms": {"voice": (0.0, 2.0)},
    "sossnare": {
        "voice": (0.0, 2.0),
        "semitone": (0.1, 2.0),  # here a frequency ratio, not semitones
        "pitch1": (500.0, 4000.0),
        "resonance": (0.0, 1.0),
    },
}

SYNTH_NAMES = frozenset(SYNTHS)

# Canonical control order for pattern text (model.to_text): core, event FX,
# synth-specific, global. Fixed so config text is deterministic.
_SPECS = (
    CORE
    + EVENT_FX
    + tuple(s for s in SYNTH_PARAM_SPECS if s.name != "resonance")
    + GLOBAL_FX
)
PARAMS: dict[str, ParamSpec] = {s.name: s for s in _SPECS}
PARAM_ORDER: tuple[str, ...] = tuple(PARAMS)

_CORE_AND_FX = frozenset(s.name for s in CORE + EVENT_FX)
_GLOBAL = frozenset(s.name for s in GLOBAL_FX)


def spec(name: str) -> ParamSpec:
    return PARAMS[name]


def synth_range(synth: str, name: str) -> tuple[float, float]:
    """Effective (lo, hi) for a synth-specific param on ``synth``."""
    override = SYNTHS[synth].get(name)
    if override is not None:
        return override
    s = PARAMS[name]
    return (s.lo, s.hi)


def effective_range(
    name: str, sources: frozenset[str] | set[str]
) -> tuple[float, float]:
    """The (lo, hi) valid for ``name`` over ``sources``: the intersection of
    the listing synths' ranges, or the base spec range if none list it."""
    s = PARAMS[name]
    listing = [sy for sy in sources & SYNTH_NAMES if name in SYNTHS[sy]]
    if not listing:
        return (s.lo, s.hi)
    return (
        max(synth_range(sy, name)[0] for sy in listing),
        min(synth_range(sy, name)[1] for sy in listing),
    )


def applicable(name: str, sources: frozenset[str] | set[str]) -> bool:
    """Is param ``name`` applicable to a config over ``sources``?

    Core, event-FX, and global params apply to any source (sample banks and
    custom synthdefs included). A synth-specific param requires every Super*
    source in the config to list it — sample banks ignore stray synth args,
    but we keep the space clean so the model never learns dead controls.
    """
    if name in _CORE_AND_FX or name in _GLOBAL:
        return True
    if name not in PARAMS:
        return False
    synths = sources & SYNTH_NAMES
    if not synths:
        return False
    return all(name in SYNTHS[s] for s in synths)


def check_value(
    name: str, value: float | str, sources: frozenset[str] | set[str]
) -> bool:
    """Range/choice check for ``name`` = ``value`` over ``sources``.

    A synth's range override applies to any param it lists (e.g. supernoise
    widens ``resonance`` to 0..1); the value must fit every listing synth in
    the config, and the base spec range otherwise.
    """
    s = PARAMS[name]
    if s.kind == CHOICE or isinstance(value, str):
        return s.in_range(value)
    if s.kind == INTEGER and value != int(value):
        return False
    listing = [sy for sy in sources & SYNTH_NAMES if name in SYNTHS[sy]]
    if listing:
        return all(
            synth_range(sy, name)[0] <= value <= synth_range(sy, name)[1]
            for sy in listing
        )
    return s.in_range(value)


def midicps(midinote: float) -> float:
    """MIDI note -> Hz (SuperDirt: freq = (note + 60).midicps at octave 5)."""
    return 440.0 * 2.0 ** ((midinote - 69.0) / 12.0)


# Continuous params that are nevertheless read only at trigger time (or
# explicitly non-modulatable: dirt_gate declares gain \ir, "gain and
# overgain can't" be modulated) — excluded from scene trajectories.
_TRIGGER_ONLY = frozenset({"gain", "speed", "accelerate", "attack", "release"})


def modulatable(name: str) -> bool:
    """May ``name`` carry a scene trajectory (grammar v3)?"""
    s = PARAMS.get(name)
    return s is not None and s.kind in (CONTINUOUS, LOG) and name not in _TRIGGER_ONLY
