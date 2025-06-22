# %%

import subprocess
import psutil
import time
import os
import signal
import sys

# ————— CONFIG —————
# How much RAM (in bytes) you allow app.py to use before pausing it.
# Here: ~6 GiB on an 8 GiB machine.
MEMORY_THRESHOLD = 6 * 1024**3  

# How long (in seconds) to sleep before resuming the process.
PAUSE_SECONDS = 1.0  

# How often to check memory (in seconds).
CHECK_INTERVAL = 0.5  


def main():
    # Launch app.py as a child process, inheriting stdout/stderr.
    proc = subprocess.Popen(
        [sys.executable, "app.py"],
        stdout=None,
        stderr=None
    )

    # Wrap the child in a psutil.Process for easy mem‐info.
    try:
        child = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        # app.py died immediately
        proc.wait()
        return proc.returncode

    # Periodically monitor the child’s RSS.
    while True:
        ret = proc.poll()
        if ret is not None:
            # app.py has exited
            return ret

        rss = child.memory_info().rss
        if rss > MEMORY_THRESHOLD:
            print(f"[memguard] RSS={rss//1024**2} MiB > "
                  f"{MEMORY_THRESHOLD//1024**2} MiB → pausing…", file=sys.stderr)
            proc.send_signal(signal.SIGSTOP)
            time.sleep(PAUSE_SECONDS)
            print("[memguard] resuming…", file=sys.stderr)
            proc.send_signal(signal.SIGCONT)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
