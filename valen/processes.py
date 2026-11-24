"""Run POSIX training commands with bounded cleanup of their descendants."""
import os
import signal
import subprocess
import sys
import time

import psutil


def run_command(cmd, *, interrupt_timeout=10.0, terminate_timeout=5.0):
    """运行命令，并在结束或中断时清理已跟踪的子孙进程。

    必须从 Python 主线程调用，因为 signal.signal 只能在主线程注册处理器。
    POSIX 下，本函数作为监督进程：启动命令、持续发现后代、接收退出信号，
    最后依次尝试原始退出信号、SIGTERM、SIGKILL，并等待各进程退出。

    interrupt_timeout：第一轮退出等待时间；未收到外部信号时用 SIGTERM。
    terminate_timeout：第二轮 SIGTERM 的等待时间；最后 SIGKILL 再等 5 秒。
    再次收到 SIGINT/SIGTERM 会请求跳过温和退出，转入 SIGKILL 清理。

    正常成功时返回 None；命令失败抛出 CalledProcessError；收到退出信号时
    抛出 SystemExit（Ctrl+C 通常为 130）；强制清理后仍存活则抛 RuntimeError。
    """
    if os.name != "posix":
        # 非 POSIX 平台沿用同步执行；这里不启用下面的进程树清理机制。
        subprocess.run(cmd, check=True)
        return

    requested = None  # 首次收到的退出信号；None 表示尚未请求停止。
    force = False     # 再次收到退出信号后置 True，要求强制清理。
    process = None    # Popen 对象：直接启动的命令，通常是 torchrun。
    tracked = set()   # 已发现的 psutil.Process 对象，包括启动器和各级后代。

    def request_stop(signum, frame):
        # 信号处理器只记状态，不在这里执行耗时清理或立即抛 KeyboardInterrupt；
        # 外层轮询会察觉 requested，并通过 finally 进入统一清理流程。
        nonlocal requested, force
        if requested is None:
            requested = signum
        else:
            force = True

    def refresh():
        # 从所有已跟踪进程继续发现后代，而不只查最初的启动器。
        # 已发现的 worker 即使换了 session，或启动器退出后被其他父进程接管，
        # 仍保留在集合中。使用快照遍历，允许发现后代时扩充 tracked。
        for child in list(tracked):
            try:
                if child.is_running():
                    tracked.update(child.children(recursive=True))
            except psutil.NoSuchProcess:
                # 查询期间进程可能恰好退出，这是正常的并发情况。
                pass
        alive = set()
        for child in tracked:
            try:
                # zombie 已停止执行，只等待父进程回收，不再向它发送信号。
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    alive.add(child)
            except psutil.NoSuchProcess:
                pass
        tracked.intersection_update(alive)  # 丢弃已退出的进程，保留存活者继续跟踪。
        return alive

    def cleanup():
        alive = refresh()
        if alive:
            print("Stopping training/evaluation processes; press Ctrl+C again to force cleanup.",
                  file=sys.stderr, flush=True)
        # 逐级升级清理力度：如 Ctrl+C 的 SIGINT -> SIGTERM -> SIGKILL。
        # 命令自行结束但还有后代存活时，第一轮直接使用 SIGTERM。
        for sig, timeout in ((requested or signal.SIGTERM, interrupt_timeout),
                             (signal.SIGTERM, terminate_timeout), (signal.SIGKILL, 5.0)):
            if force:
                sig, timeout = signal.SIGKILL, 5.0
            deadline = time.monotonic() + timeout  # 单调时钟不受系统时间调整影响。
            signaled = set()  # 每轮每个进程只发一次信号；新发现的后代仍会收到。
            while alive:
                # 启动器排在最后发送，尽量让它有机会回收 worker 的退出状态。
                for child in sorted(alive - signaled, key=lambda p: p.pid == process.pid):
                    try:
                        # psutil 会检查进程身份（PID 与创建时间），避免 PID 被复用后
                        # 误向无关进程发送信号；比直接 os.kill(pid, sig) 更稳妥。
                        child.send_signal(sig)
                    except psutil.NoSuchProcess:
                        pass
                    signaled.add(child)
                process.poll()  # 非阻塞检查并回收直接子进程，更新 returncode。
                alive = refresh()  # 清理期间仍继续发现新后代，并移除已退出者。
                if time.monotonic() >= deadline or (force and sig != signal.SIGKILL):
                    # 等待超时，或再次 Ctrl+C：结束当前轮，升级到下一轮清理。
                    break
                if alive:
                    time.sleep(0.1)
            if not alive:
                break
        process.poll()
        if alive:
            raise RuntimeError(f"Processes still exiting after SIGKILL: {sorted(p.pid for p in alive)}")

    # 暂存调用方的处理器；函数退出时恢复，避免影响后续训练阶段或其他代码。
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in handlers:
            signal.signal(sig, request_stop)
        if requested is None:
            # 将命令放入新 session，使终端 Ctrl+C 先由当前监督进程统一处理。
            # torchrun 的 worker 还可能自行创建 session，故不能只杀一个进程组，
            # 而要持续跟踪并清理具体的后代进程。
            process = subprocess.Popen(cmd, start_new_session=True)
            print(f"Run cmd: {' '.join(map(str, cmd))} (PID {process.pid})", flush=True)
            try:
                tracked.add(psutil.Process(process.pid))
            except psutil.NoSuchProcess:
                pass
            while True:
                refresh()
                if requested is not None or process.poll() is not None:
                    # 用户请求停止，或命令已退出（成功/失败均可），都结束监督轮询。
                    break
                time.sleep(0.1)
    finally:
        # 正常结束、中断、执行异常都会走这里，清理仍然存活的已知后代。
        try:
            if process is not None:
                cleanup()
        finally:
            # 即使 cleanup 自身报错，也必须恢复调用方原有的信号处理器。
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    if requested is not None:
        # 用常见的 shell 退出码约定传播中断：SIGINT=2 -> 130，SIGTERM=15 -> 143。
        raise SystemExit(128 + requested)
    if process.returncode:
        # 没有外部中断时，保留 subprocess.run(check=True) 的失败传播语义。
        raise subprocess.CalledProcessError(process.returncode, cmd)
