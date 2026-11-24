"""Opt-in per-rank phase traces and Python stacks for stalled training steps."""
from collections import Counter
from contextlib import contextmanager
import faulthandler
import json
import math
import os
import sys
import time


def trace_options():
    steps = int(os.environ.get("VALEN_TRACE_STEPS", "0"))
    timeout = float(os.environ.get("VALEN_TRACE_TIMEOUT", "60"))
    if steps < 0 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("VALEN_TRACE_STEPS must be nonnegative; VALEN_TRACE_TIMEOUT must be positive")
    return steps, timeout


class PhaseTrace:
    """No GPU operations or distributed collectives; each worker reports locally.

    The optional faulthandler timer prints Python stacks, not native CUDA stacks,
    and never terminates training. Only enable it when no other code owns the
    process-wide dump_traceback_later timer.
    """

    def __init__(self, rank, step, *, enabled=False, timeout=60):
        self.rank, self.step = rank, step
        self.enabled, self.timeout = enabled, timeout
        self.calls = Counter()

    def emit(self, phase, status, **fields):
        if self.enabled:
            print(json.dumps({"event": "training_phase", "rank": self.rank, "pid": os.getpid(),
                "step": self.step, "phase": phase, "call": self.calls[phase],
                "status": status, **fields}), file=sys.stderr, flush=True)

    @contextmanager
    def phase(self, name):
        if not self.enabled:
            yield
            return
        self.calls[name] += 1
        started = time.monotonic()
        self.emit(name, "begin")
        faulthandler.dump_traceback_later(self.timeout, repeat=True, file=sys.stderr, exit=False)
        try:
            yield
        except BaseException as exc:
            self.emit(name, "error", error_type=type(exc).__name__, error=str(exc))
            raise
        else:
            self.emit(name, "end", elapsed_seconds=time.monotonic() - started)
        finally:
            faulthandler.cancel_dump_traceback_later()
