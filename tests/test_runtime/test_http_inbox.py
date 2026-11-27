"""Private-network inbox relay contracts."""

from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import unittest
from unittest import mock
from urllib.request import Request, urlopen

from specforge.runtime.contracts import FeatureSpec, SampleRef
from specforge.runtime.data_plane.http_inbox import (
    InboxHTTPServer,
    InboxHTTPServerProcess,
    RemoteInboxChannel,
)
from specforge.runtime.data_plane.ref_distributor import RefDistributor
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel


def _ref(sample_id: str) -> SampleRef:
    return SampleRef(
        sample_id=sample_id,
        run_id="run0",
        source_task_id=f"task-{sample_id}",
        feature_store_uri=f"mooncake://run0/{sample_id}",
        feature_keys={"hidden_state": f"{sample_id}/hidden_state"},
        feature_specs={
            "hidden_state": FeatureSpec(
                name="hidden_state", shape=(2, 4), dtype="float32"
            )
        },
        strategy="dspark",
        metadata={"target_repr": "hidden_state"},
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TestHTTPInbox(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.mkdtemp(prefix="http-inbox-")
        self.path = RefDistributor.inbox_path(self.work, 1)
        self.local = StreamingRefChannel(self.path)
        self.origin = f"http://127.0.0.1:{_free_port()}"
        self.server = InboxHTTPServer(
            self.work, 2, self.origin, bind_host="127.0.0.1"
        ).start()
        self.remote = RemoteInboxChannel(self.origin, 1)

    def tearDown(self):
        self.server.stop()

    def test_tail_read_close_and_consumed_counter(self):
        self.local.publish_batch([_ref("s0"), _ref("s1")])
        self.assertEqual([ref.sample_id for ref in self.remote.poll()], ["s0", "s1"])
        self.assertEqual(self.remote.poll(), [])

        self.remote.mark_consumed(2)
        self.assertEqual(self.local.consumed_remote(), 2)

        self.remote.mark_consumed(1)
        self.assertEqual(self.local.consumed_remote(), 3)

        self.local.close()
        self.assertTrue(self.remote.is_closed())

    def test_status_probe_does_not_discard_unpolled_refs(self):
        self.local.publish(_ref("s0"))
        self.assertFalse(self.remote.is_closed())
        self.assertIsNone(self.remote.failure())
        self.assertEqual([ref.sample_id for ref in self.remote.poll()], ["s0"])

    def test_failure_is_forwarded(self):
        failure = self.path + ".failed"
        with open(failure, "w", encoding="utf-8") as stream:
            stream.write("capture failed")
        self.assertIn("capture failed", self.remote.failure())

    def test_consumed_target_is_idempotent_under_concurrent_retries(self):
        body = json.dumps({"target": 5}).encode("utf-8")

        def post_target():
            request = Request(
                f"{self.origin}/v1/inboxes/1/consumed",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=2.0) as response:
                self.assertEqual(json.load(response)["consumed"], 5)

        threads = [threading.Thread(target=post_target) for _ in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(self.local.consumed_remote(), 5)

    def test_concurrent_client_acks_accumulate_exactly(self):
        # The background durable ack and loader failure settlement can both
        # advance one client's target; neither update may be lost.
        errors = []

        def ack_many():
            try:
                for _ in range(25):
                    self.remote.mark_consumed(1)
            except BaseException as exc:  # surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=ack_many) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(self.local.consumed_remote(), 150)

    def test_pull_treats_connection_reset_as_transient(self):
        with mock.patch(
            "specforge.runtime.data_plane.http_inbox.urlopen",
            side_effect=ConnectionResetError("peer reset"),
        ):
            self.assertEqual(self.remote.poll(), [])

    def test_invalid_origin_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "http://host:port"):
            RemoteInboxChannel("https://trainer.example:35900/path", 1)


class TestHTTPInboxProcess(TestHTTPInbox):
    """The same relay contract when the server lives in its own process."""

    def setUp(self):
        self.work = tempfile.mkdtemp(prefix="http-inbox-proc-")
        self.path = RefDistributor.inbox_path(self.work, 1)
        self.local = StreamingRefChannel(self.path)
        self.origin = f"http://127.0.0.1:{_free_port()}"
        self.server = InboxHTTPServerProcess(
            self.work, 2, self.origin, bind_host="127.0.0.1"
        ).start()
        self.remote = RemoteInboxChannel(self.origin, 1)

    def test_server_runs_in_a_separate_process_and_stops(self):
        proc = self.server._proc
        self.assertIsNotNone(proc)
        self.assertNotEqual(proc.pid, os.getpid())
        self.assertIsNone(proc.poll())
        self.server.stop()
        self.assertIsNotNone(proc.poll())
        self.assertIsNone(self.server._proc)
        # a second stop is a no-op, and the port is released
        self.server.stop()
        with self.assertRaises(OSError):
            urlopen(f"{self.origin}/v1/inboxes/0?offset=0", timeout=1.0)

    def test_startup_failure_is_reported(self):
        # an address the child cannot bind (TEST-NET-3) makes it exit at startup;
        # the parent surfaces the child's error instead of waiting for the timeout
        origin = f"http://127.0.0.1:{_free_port()}"
        with self.assertRaisesRegex(RuntimeError, "exited at startup"):
            InboxHTTPServerProcess(
                self.work, 2, origin, bind_host="203.0.113.1", startup_timeout_s=20.0
            ).start()


if __name__ == "__main__":
    unittest.main()
