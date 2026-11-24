"""Exercise real subprocesses, including torchrun-style separate sessions."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil
import pytest

from valen.processes import run_command


pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX signal supervision")
ROOT = Path(__file__).resolve().parents[2]


def wait_until(condition, timeout=10):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            pytest.fail("Timed out waiting for subprocess state")
        time.sleep(.05)


def alive(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def test_command_success_failure_and_signal_handlers():
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    run_command([sys.executable, "-c", "pass"])
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_command([sys.executable, "-c", "raise SystemExit(7)"])
    assert error.value.returncode == 7
    with pytest.raises(FileNotFoundError):
        run_command(["/nonexistent/valen-test-command"])
    assert handlers == {sig: signal.getsignal(sig) for sig in handlers}


@pytest.mark.parametrize("signum,repeat", [(signal.SIGINT, False), (signal.SIGTERM, False),
                                         (signal.SIGINT, True)])
def test_interrupt_cleans_detached_workers_after_launcher_exits(tmp_path, signum, repeat):
    # Both worker levels ignore graceful signals and create separate sessions.
    # The launcher exits promptly, leaving the supervisor to clean its orphans.
    grandchild = tmp_path / "grandchild.py"
    grandchild.write_text("""
import os, signal, time
from pathlib import Path
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(__file__).with_suffix('.ready').write_text(str(os.getpid()))
while True: time.sleep(.1)
""")
    worker = tmp_path / "worker.py"
    worker.write_text(f"""
import json, os, signal, subprocess, sys, time
from pathlib import Path
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, {str(grandchild)!r}], start_new_session=True)
Path(__file__).with_suffix('.ready').write_text(json.dumps([os.getpid(), child.pid]))
while True: time.sleep(.1)
""")
    launcher = tmp_path / "launcher.py"
    launcher.write_text(f"""
import os, signal, subprocess, sys, time
from pathlib import Path
def stop(sig, frame): raise SystemExit(0)
signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
child = subprocess.Popen([sys.executable, {str(worker)!r}], start_new_session=True)
Path(__file__).with_suffix('.ready').write_text(str(os.getpid()))
while True: time.sleep(.1)
""")
    supervisor_code = (
        "from valen.processes import run_command; "
        f"run_command({[sys.executable, str(launcher)]!r}, "
        f"interrupt_timeout={30 if repeat else .4}, terminate_timeout=.4)"
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                 start_new_session=True)
    processes = []
    log = tmp_path / "stderr.log"
    with log.open("w") as stderr:
        supervisor = subprocess.Popen([sys.executable, "-c", supervisor_code], cwd=ROOT,
                                      start_new_session=True, stderr=stderr)
        try:
            wait_until(lambda: grandchild.with_suffix(".ready").exists()
                       and worker.with_suffix(".ready").exists()
                       and launcher.with_suffix(".ready").exists())
            pids = [int(launcher.with_suffix(".ready").read_text())]
            pids += json.loads(worker.with_suffix(".ready").read_text())
            processes = [psutil.Process(pid) for pid in pids]
            assert len({os.getsid(pid) for pid in pids}) == 3
            supervisor.send_signal(signum)
            if repeat:
                wait_until(lambda: "Stopping training/evaluation" in log.read_text())
                supervisor.send_signal(signal.SIGINT)
            assert supervisor.wait(timeout=10) == 128 + signum
            wait_until(lambda: not any(alive(p) for p in processes))
            assert unrelated.poll() is None
        finally:
            # Also clean fixtures if an assertion fails before PID collection.
            for path in (launcher, worker, grandchild):
                ready = path.with_suffix(".ready")
                if ready.exists():
                    values = json.loads(ready.read_text())
                    for pid in values if isinstance(values, list) else [values]:
                        try:
                            processes.append(psutil.Process(pid))
                        except psutil.NoSuchProcess:
                            pass
            for proc in processes:
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
            for proc in (supervisor, unrelated):
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=5)
