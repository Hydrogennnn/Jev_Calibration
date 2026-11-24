import io
import json
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from valen.training.diagnostics import PhaseTrace, trace_options


class PhaseTraceTests(unittest.TestCase):
    def test_disabled_does_not_touch_process_watchdog(self):
        stream = io.StringIO()
        with patch("valen.training.diagnostics.faulthandler.dump_traceback_later") as arm, \
             patch("valen.training.diagnostics.faulthandler.cancel_dump_traceback_later") as cancel, \
             redirect_stderr(stream):
            trace = PhaseTrace(1, 1)
            with trace.phase("forward"):
                trace.emit("forward", "body")
        self.assertEqual(stream.getvalue(), "")
        arm.assert_not_called()
        cancel.assert_not_called()

    def test_completed_and_failed_phases_are_distinguishable_and_cancel_watchdog(self):
        stream = io.StringIO()
        with patch("valen.training.diagnostics.faulthandler.dump_traceback_later") as arm, \
             patch("valen.training.diagnostics.faulthandler.cancel_dump_traceback_later") as cancel, \
             redirect_stderr(stream):
            trace = PhaseTrace(1, 3, enabled=True, timeout=20)
            with trace.phase("forward"):
                trace.emit("forward", "body")
            with self.assertRaisesRegex(RuntimeError, "failed"):
                with trace.phase("forward"):
                    raise RuntimeError("failed")
        events = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual([e["status"] for e in events], ["begin", "body", "end", "begin", "error"])
        self.assertEqual([e["call"] for e in events], [1, 1, 1, 2, 2])
        self.assertTrue(all(e["rank"] == 1 and e["step"] == 3 for e in events))
        self.assertGreaterEqual(events[2]["elapsed_seconds"], 0)
        self.assertEqual(events[-1]["error_type"], "RuntimeError")
        self.assertEqual(arm.call_count, 2)
        self.assertEqual(cancel.call_count, 2)
        arm.assert_called_with(20, repeat=True, file=stream, exit=False)

    def test_environment_options(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(trace_options(), (0, 60))
        for values in ({"VALEN_TRACE_STEPS": "-1"}, {"VALEN_TRACE_TIMEOUT": "nan"},
                       {"VALEN_TRACE_TIMEOUT": "0"}):
            with patch.dict("os.environ", values, clear=True), self.assertRaises(ValueError):
                trace_options()


if __name__ == "__main__":
    unittest.main()
