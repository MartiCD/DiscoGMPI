#!/usr/bin/env python3
"""Run an MPI launcher; terminate its process group if progress stops.

Uses only the Python standard library. The heartbeat must be updated after
completed timesteps, never by an independent timer inside the solver.
"""
import argparse
import math
from pathlib import Path
import os
import signal
import subprocess
import sys
import time


def terminate_group(process, grace):
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            continue
        # Even if the launcher exited, kill any surviving local descendants.
        if sig == signal.SIGTERM:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heartbeat', type=Path, required=True)
    parser.add_argument('--stall-seconds', type=float, default=2700)
    parser.add_argument('--startup-seconds', type=float, default=7200)
    parser.add_argument('--poll-seconds', type=float, default=5)
    parser.add_argument('--terminate-grace', type=float, default=30)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or any(not math.isfinite(x) or x <= 0 for x in (
            args.stall_seconds,args.startup_seconds,args.poll_seconds,args.terminate_grace)):
        parser.error('Provide a command and positive finite timeout settings')
    if args.heartbeat.exists():
        parser.error('Heartbeat file already exists; use a unique path for each launch')
    process = subprocess.Popen(command, start_new_session=True)
    last_progress = time.monotonic()
    last_stamp = None
    try:
        while True:
            code = process.poll()
            if code is not None:
                return code if code >= 0 else 128-code
            try:
                stamp = args.heartbeat.stat().st_mtime_ns
            except FileNotFoundError:
                stamp = None
            if stamp is not None and stamp != last_stamp:
                last_stamp = stamp
                last_progress = time.monotonic()
            timeout = args.startup_seconds if last_stamp is None else args.stall_seconds
            if time.monotonic()-last_progress > timeout:
                phase = 'startup' if last_stamp is None else 'timestep/checkpoint/finalization'
                print(f'WATCHDOG_TIMEOUT: no {phase} progress for {timeout:g}s; '
                      'terminating MPI launcher and retaining committed checkpoints',
                      file=sys.stderr, flush=True)
                terminate_group(process,args.terminate_grace)
                return 124
            time.sleep(args.poll_seconds)
    finally:
        if process.poll() is None:
            terminate_group(process,args.terminate_grace)


if __name__ == '__main__':
    sys.exit(main())
