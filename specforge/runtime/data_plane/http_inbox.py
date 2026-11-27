# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Private-network HTTP transport for rank-local online reference inboxes.

Online feature tensors remain in Mooncake.  This module relays only the small,
tensor-free ``SampleRef`` JSONL records that :class:`RefDistributor` already
writes.  It lets multi-node consumers run on container platforms where trainer
nodes cannot mount a shared control filesystem.

Rank 0 owns the server and its local inbox files.  Every other rank tail-reads
its private stream by byte offset and posts only its durable consumed-count
target.  Requests are idempotent with respect to a client retry: a read offset
is advanced only after the response arrives, and an absolute consumed target is
applied under a per-rank server lock.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from specforge.runtime.contracts import SampleRef
from specforge.runtime.data_plane.ref_distributor import RefDistributor
from specforge.runtime.data_plane.ref_serialization import ref_from_dict
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel

_CLOSED_SUFFIX = ".closed"
_FAILED_SUFFIX = ".failed"


class _InboxThreadingHTTPServer(ThreadingHTTPServer):
    # socketserver.TCPServer defaults to a backlog of five.  A DP consumer can
    # have dozens of ranks polling in lockstep, so that default turns a healthy
    # rank-0 relay into intermittent connection resets at every step boundary.
    request_queue_size = 256
    daemon_threads = True


def _validated_origin(origin: str):
    parsed = urlparse(origin)
    if (
        parsed.scheme != "http"
        or not parsed.hostname
        or parsed.port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "inbox HTTP origin must be http://host:port without credentials, "
            "path, query, or fragment"
        )
    return parsed


class InboxHTTPServer:
    """Serve rank-0 inbox files and consumed counters on a private origin."""

    def __init__(
        self,
        inbox_dir: str,
        dp_size: int,
        origin: str,
        *,
        bind_host: str = "0.0.0.0",
    ) -> None:
        parsed = _validated_origin(origin)
        if dp_size < 2:
            raise ValueError("inbox HTTP server requires dp_size >= 2")
        self.inbox_dir = os.path.abspath(inbox_dir)
        self.dp_size = dp_size
        self.origin = origin.rstrip("/")
        self._ack_channels = [
            StreamingRefChannel(RefDistributor.inbox_path(self.inbox_dir, rank))
            for rank in range(dp_size)
        ]
        # A target ack may be retried after an ambiguous connection reset.  The
        # per-rank lock makes the read/advance/respond sequence atomic across
        # the original request and its retry.
        self._ack_locks = [threading.Lock() for _ in range(dp_size)]
        self._httpd = _InboxThreadingHTTPServer(
            (bind_host, parsed.port), self._handler_type()
        )
        self._thread: threading.Thread | None = None

    def _handler_type(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "SpecForgeInbox/1"

            def log_message(self, _format, *_args):
                return

            def _rank(self, *, consumed: bool = False) -> int | None:
                suffix = "/consumed" if consumed else ""
                path = urlparse(self.path).path
                prefix = "/v1/inboxes/"
                if not path.startswith(prefix) or not path.endswith(suffix):
                    return None
                token = path[len(prefix) :]
                if suffix:
                    token = token[: -len(suffix)]
                try:
                    rank = int(token)
                except ValueError:
                    return None
                return rank if 0 <= rank < owner.dp_size else None

            def _json(self, status: HTTPStatus, payload) -> None:
                body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                rank = self._rank()
                if rank is None:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "unknown inbox"})
                    return
                query = parse_qs(urlparse(self.path).query)
                try:
                    offset = int(query.get("offset", ["0"])[0])
                except ValueError:
                    offset = -1
                if offset < 0:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid offset"})
                    return
                path = RefDistributor.inbox_path(owner.inbox_dir, rank)
                data = b""
                next_offset = offset
                try:
                    with open(path, "rb") as stream:
                        stream.seek(offset)
                        data = stream.read()
                        next_offset = stream.tell()
                except FileNotFoundError:
                    pass
                failure = None
                try:
                    with open(path + _FAILED_SUFFIX, encoding="utf-8") as stream:
                        failure = stream.read()
                except FileNotFoundError:
                    pass
                self._json(
                    HTTPStatus.OK,
                    {
                        "data": base64.b64encode(data).decode("ascii"),
                        "next_offset": next_offset,
                        "closed": os.path.exists(path + _CLOSED_SUFFIX),
                        "failure": failure,
                    },
                )

            def do_POST(self):
                rank = self._rank(consumed=True)
                if rank is None:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "unknown inbox"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length < 1 or length > 1024:
                        raise ValueError("invalid body size")
                    payload = json.loads(self.rfile.read(length))
                    target = int(payload["target"])
                    if target < 1:
                        raise ValueError("target must be positive")
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                with owner._ack_locks[rank]:
                    channel = owner._ack_channels[rank]
                    current = channel.consumed_remote()
                    if target < current:
                        self._json(
                            HTTPStatus.CONFLICT,
                            {
                                "error": "consumed target moved backwards",
                                "consumed": current,
                            },
                        )
                        return
                    if target > current:
                        channel.mark_consumed(target - current)
                    self._json(HTTPStatus.OK, {"consumed": target})

        return Handler

    def start(self) -> InboxHTTPServer:
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="specforge-inbox-http",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5.0)
        self._thread = None


class InboxHTTPServerProcess:
    """``InboxHTTPServer`` hosted in its own interpreter process.

    Rank 0 normally serves the inbox from a thread inside the trainer.  With a
    very small draft (FFN <= 768) the trainer's compiled backward keeps the
    GIL busy (``cuLaunchKernel`` spins while the launch queue drains behind a
    pending NCCL kernel), the serving thread starves, every other rank times
    out on its ref pulls and the collective never completes.  The relay only
    reads and writes files in ``inbox_dir``, so moving it out of the trainer
    process removes the coupling without changing the protocol.
    """

    def __init__(
        self,
        inbox_dir: str,
        dp_size: int,
        origin: str,
        *,
        bind_host: str = "0.0.0.0",
        startup_timeout_s: float = 60.0,
    ) -> None:
        parsed = _validated_origin(origin)
        if dp_size < 2:
            raise ValueError("inbox HTTP server requires dp_size >= 2")
        self.inbox_dir = os.path.abspath(inbox_dir)
        self.dp_size = dp_size
        self.origin = origin.rstrip("/")
        self.bind_host = bind_host
        self.port = parsed.port
        self.startup_timeout_s = startup_timeout_s
        self._proc: subprocess.Popen | None = None

    def _probe_origin(self) -> str:
        host = "127.0.0.1" if self.bind_host in ("", "0.0.0.0", "::") else self.bind_host
        return f"http://{host}:{self.port}"

    def start(self) -> InboxHTTPServerProcess:
        if self._proc is not None:
            return self
        cmd = [
            sys.executable,
            "-m",
            "specforge.runtime.data_plane.http_inbox",
            "--inbox-dir",
            self.inbox_dir,
            "--dp-size",
            str(self.dp_size),
            "--origin",
            self.origin,
            "--bind-host",
            self.bind_host,
        ]
        # The child must import ``specforge`` the same way this process did
        # (an editable checkout on PYTHONPATH is the common case on the pods).
        import specforge

        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(specforge.__file__)))
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [pkg_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
        )
        deadline = time.monotonic() + self.startup_timeout_s
        probe = Request(f"{self._probe_origin()}/v1/inboxes/0?offset=0")
        while True:
            if self._proc.poll() is not None:
                err = (self._proc.stderr.read() if self._proc.stderr else b"").decode(
                    "utf-8", "replace"
                )[-2000:]
                self._proc = None
                raise RuntimeError(f"inbox HTTP server process exited at startup: {err}")
            try:
                with urlopen(probe, timeout=2.0) as response:
                    if response.status == 200:
                        return self
            except (URLError, OSError, TimeoutError):
                pass
            if time.monotonic() > deadline:
                self.stop()
                raise RuntimeError(
                    f"inbox HTTP server process did not answer on {self.origin} "
                    f"within {self.startup_timeout_s:.0f}s"
                )
            time.sleep(0.2)

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
        if proc.stderr is not None:
            proc.stderr.close()


def main(argv: list[str] | None = None) -> int:
    """Serve one inbox directory until SIGTERM/SIGINT (used by ``InboxHTTPServerProcess``)."""
    parser = argparse.ArgumentParser(prog="specforge.runtime.data_plane.http_inbox")
    parser.add_argument("--inbox-dir", required=True)
    parser.add_argument("--dp-size", type=int, required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--bind-host", default="0.0.0.0")
    args = parser.parse_args(argv)
    server = InboxHTTPServer(
        args.inbox_dir, args.dp_size, args.origin, bind_host=args.bind_host
    )
    httpd = server._httpd
    stop_requested = threading.Event()

    def _stop(_signum, _frame):
        if not stop_requested.is_set():
            stop_requested.set()
            threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
    return 0


class RemoteInboxChannel:
    """Read one rank inbox and acknowledge it through ``InboxHTTPServer``."""

    def __init__(self, origin: str, dp_rank: int, *, timeout_s: float = 10.0) -> None:
        _validated_origin(origin)
        if dp_rank < 0:
            raise ValueError("dp_rank must be non-negative")
        self.path = f"{origin.rstrip('/')}/v1/inboxes/{dp_rank}"
        self.timeout_s = timeout_s
        self._read_offset = 0
        self._buf = ""
        self._closed = False
        self._failure: str | None = None
        self._pending: list[SampleRef] = []
        self._pull_lock = threading.Lock()
        # The durable ack (possibly on the background ack thread) and loader
        # failure settlement both advance the target; a lost update silently
        # undercounts it, or posts a target that moved backwards (rejected).
        self._consumed_lock = threading.Lock()
        self._consumed_target = 0

    def _pull(self) -> None:
        with self._pull_lock:
            request = Request(f"{self.path}?offset={self._read_offset}")
            try:
                with urlopen(request, timeout=self.timeout_s) as response:
                    payload = json.load(response)
            except (URLError, OSError, TimeoutError):
                # Let StreamingRefQueue's configured idle timeout distinguish a
                # transient network outage from a dead rank-0 service.
                return
            data = base64.b64decode(payload["data"])
            next_offset = int(payload["next_offset"])
            if (
                next_offset < self._read_offset
                or next_offset - self._read_offset != len(data)
            ):
                raise RuntimeError("inbox HTTP server returned an invalid byte range")
            self._read_offset = next_offset
            self._closed = bool(payload["closed"])
            self._failure = payload.get("failure")
            self._buf += data.decode("utf-8")
            lines = self._buf.split("\n")
            self._buf = lines.pop()
            self._pending.extend(
                ref_from_dict(json.loads(line)) for line in lines if line
            )

    def poll(self) -> list[SampleRef]:
        self._pull()
        refs, self._pending = self._pending, []
        return refs

    def is_closed(self) -> bool:
        self._pull()
        return self._closed

    def failure(self) -> str | None:
        self._pull()
        if self._failure is None:
            return None
        return f"ref-distributor died:\n{self._failure}"

    def mark_consumed(self, n: int) -> None:
        if n < 1:
            return
        with self._consumed_lock:
            self._mark_consumed_locked(n)

    def _mark_consumed_locked(self, n: int) -> None:
        target = self._consumed_target + n
        body = json.dumps({"target": target}, separators=(",", ":")).encode("utf-8")
        transient_error = None
        for attempt in range(4):
            request = Request(
                f"{self.path}/consumed",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout_s) as response:
                    payload = json.load(response)
            except HTTPError:
                raise
            except (URLError, OSError, TimeoutError) as exc:
                transient_error = exc
                if attempt < 3:
                    time.sleep(0.05 * (attempt + 1))
                continue
            if int(payload.get("consumed", -1)) != target:
                raise RuntimeError("inbox HTTP server did not confirm consumed target")
            self._consumed_target = target
            return
        raise RuntimeError(
            "inbox HTTP consumed acknowledgement failed after retries"
        ) from transient_error


__all__ = ["InboxHTTPServer", "InboxHTTPServerProcess", "RemoteInboxChannel", "main"]


if __name__ == "__main__":  # pragma: no cover - exercised through InboxHTTPServerProcess
    sys.exit(main())
