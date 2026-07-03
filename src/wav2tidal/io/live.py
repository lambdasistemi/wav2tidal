"""The live scene player: one resident SuperDirt session (US3-2, #51).

Boots sclang+SuperDirt once on dedicated ports (fleet-style, so it
coexists with dataset rendering), then swaps playing scenes on command:
``play(plan)`` writes a generated sclang chunk (``build_live_swap_chunk``)
and tells the resident script to load it over OSC. Free swapping per the
US3 design: the new scene starts, the old groups are freed — a hard cut.

By default the session plays to the system output (that is the point);
pass ``sink`` to route into a PipeWire null sink for tests, and capture
its ``.monitor`` with ``pw-record`` — the same capture path the pursuit
loop's verify step uses.
"""

from __future__ import annotations

import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path

from ..core.osc import message
from .superdirt import (
    _make_null_sink,
    _resolve_sclang,
    _unload_null_sink,
    build_live_boot_script,
    build_live_swap_chunk,
)


class LiveSession:
    """Own the resident sclang process and command it over OSC."""

    def __init__(
        self,
        *,
        banks_dir: str | Path | None = None,
        dirt_port: int = 57360,
        server_port: int = 57160,
        lang_port: int = 59360,
        sink: str | None = None,
        sclang: str | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        self.dirt_port = dirt_port
        self.server_port = server_port
        self.lang_port = lang_port
        self.sink = sink
        self._banks_dir = str(banks_dir) if banks_dir else None
        self._sclang = _resolve_sclang(sclang)
        self._tmp = tempfile.TemporaryDirectory(prefix="w2t_live_")
        self._log = Path(log_path) if log_path else Path(self._tmp.name) / "live.log"
        self._proc: subprocess.Popen | None = None
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sink_id: str | None = None
        self._n_chunks = 0

    # -- lifecycle -----------------------------------------------------------

    def start(self, timeout: float = 90.0) -> None:
        script = Path(self._tmp.name) / "boot.scd"
        script.write_text(
            build_live_boot_script(
                port=self.dirt_port,
                server_port=self.server_port,
                banks_dir=self._banks_dir,
            )
        )
        env = dict(os.environ)
        if self.sink:
            self._sink_id = _make_null_sink(self.sink)
            if self._sink_id:
                env["SC_JACK_DEFAULT_OUTPUTS"] = (
                    f"{self.sink}:playback_FL,{self.sink}:playback_FR"
                )
        log = open(self._log, "w")
        self._proc = subprocess.Popen(
            [self._sclang, "-u", str(self.lang_port), str(script)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            self._wait_for("W2T_LIVE_READY", timeout)
            if self.sink:
                self._link_outputs()
        except Exception:
            # a failed start must never leak the process (it holds the
            # SuperDirt UDP port hostage for every later session)
            self.stop(timeout=2.0)
            raise

    def _link_outputs(self) -> None:
        """Wire OUR scsynth's outputs to the sink by PipeWire port id.

        Name-based linking is ambiguous: every scsynth appears as plain
        ``SuperCollider:out_*`` in the graph (render fleets included), so
        we resolve our own instance through its process id via pw-dump.
        """
        import json

        deadline = time.monotonic() + 20.0
        last_err = "scsynth not found"
        while time.monotonic() < deadline:
            # our scsynth is unambiguous by its unique -u server port
            pids = subprocess.run(
                ["pgrep", "-f", f"scsynth -u {self.server_port} "],
                capture_output=True,
                text=True,
            ).stdout.split()
            if not pids:
                time.sleep(0.25)
                continue
            scsynth_pid = int(pids[0])
            dump = json.loads(
                subprocess.run(["pw-dump"], capture_output=True, text=True).stdout
            )
            # the process id lives on the Client object; the Node carries
            # client.id; the Ports carry node.id — walk the chain
            client_id = None
            for obj in dump:
                if not obj.get("type", "").endswith("Client"):
                    continue
                props = obj.get("info", {}).get("props", {})
                if props.get("application.process.id") == scsynth_pid:
                    client_id = obj["id"]
                    break
            node_id = None
            if client_id is not None:
                for obj in dump:
                    if not obj.get("type", "").endswith("Node"):
                        continue
                    props = obj.get("info", {}).get("props", {})
                    if props.get("client.id") == client_id:
                        node_id = obj["id"]
                        break
            if node_id is None:
                last_err = f"no PipeWire node for scsynth pid {scsynth_pid}"
                time.sleep(0.25)
                continue
            ports = {}
            for obj in dump:
                if not obj.get("type", "").endswith("Port"):
                    continue
                props = obj.get("info", {}).get("props", {})
                if (
                    props.get("node.id") == node_id
                    and props.get("port.direction") == "out"
                ):
                    ports[props.get("port.name")] = obj["id"]
            if "out_1" not in ports or "out_2" not in ports:
                last_err = f"scsynth ports not ready: {sorted(ports)}"
                time.sleep(0.25)
                continue
            for pname, sink_port in (
                ("out_1", f"{self.sink}:playback_FL"),
                ("out_2", f"{self.sink}:playback_FR"),
            ):
                r = subprocess.run(
                    ["pw-link", str(ports[pname]), sink_port],
                    capture_output=True,
                    text=True,
                )
                if r.returncode != 0 and "exist" not in (r.stderr or ""):
                    raise RuntimeError(
                        f"pw-link {ports[pname]} -> {sink_port}: {r.stderr}"
                    )
            return
        raise RuntimeError(f"could not link session outputs: {last_err}")

    def stop(self, timeout: float = 10.0) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._send("/w2t/quit")
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(self._proc.pid, 9)
                self._proc.wait()
        self._proc = None
        if self._sink_id:
            _unload_null_sink(self._sink_id)
            self._sink_id = None

    def __enter__(self) -> LiveSession:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- commands ------------------------------------------------------------

    def play(self, plan, timeout: float = 15.0) -> None:
        """Swap the playing scene to ``plan`` (a dirt.ScenePlan)."""
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("live session is not running")
        self._n_chunks += 1
        chunk = Path(self._tmp.name) / f"chunk_{self._n_chunks:04d}.scd"
        chunk.write_text(build_live_swap_chunk(plan, port=self.dirt_port))
        self._send("/w2t/load", str(chunk))
        self._wait_for(f"W2T_LOADED {chunk}", timeout)

    # -- internals -----------------------------------------------------------

    def _send(self, address: str, *args) -> None:
        self._sock.sendto(message(address, *args), ("127.0.0.1", self.lang_port))

    def _wait_for(self, sentinel: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"live session died:\n{self._log.read_text()[-800:]}"
                )
            if sentinel in self._log.read_text():
                return
            time.sleep(0.1)
        raise RuntimeError(
            f"live session: no {sentinel!r} within {timeout:.0f}s"
            f"\n{self._log.read_text()[-800:]}"
        )
