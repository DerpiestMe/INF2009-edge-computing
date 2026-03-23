import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puppypi_movement import PuppyPiMovementController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive IJKL PuppyPi movement test")
    parser.add_argument("--speed", type=float, default=10.0, help="Linear speed in cm/s for I/K")
    parser.add_argument("--yaw-deg", type=float, default=25.0, help="Yaw rate in deg/s for J/L")
    parser.add_argument("--repeat-timeout", type=float, default=0.25, help="Auto-stop if no key repeat in this many seconds")
    parser.add_argument("--loop-hz", type=float, default=30.0, help="Control loop frequency")
    return parser.parse_args()


def _read_key_nonblocking_posix() -> Optional[str]:
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
        if rlist:
            return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return None


def _read_key_nonblocking_windows() -> Optional[str]:
    import msvcrt

    if msvcrt.kbhit():
        ch = msvcrt.getwch()
        return ch
    return None


def _read_key_nonblocking() -> Optional[str]:
    if os.name == "nt":
        return _read_key_nonblocking_windows()
    return _read_key_nonblocking_posix()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    yaw_rate = args.yaw_deg * (3.141592653589793 / 180.0)
    dt = 1.0 / max(1.0, args.loop_hz)

    controller = PuppyPiMovementController(
        max_x_cm_s=max(5.0, args.speed),
        max_yaw_rate_rad_s=max(0.2, yaw_rate),
    )
    if not controller.start():
        print("[ERROR] Movement controller failed to initialize (ROS stack unavailable).")
        return

    running = True
    active_cmd_ts = 0.0
    active_x = 0.0
    active_yaw = 0.0

    def _shutdown(_signum: int, _frame) -> None:
        nonlocal running
        running = False
        controller.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print("IJKL movement test started.")
    print("Keys: i=forward, k=backward, j=turn-left, l=turn-right, space=stop, h=go_home, q=quit")

    try:
        while running:
            key = _read_key_nonblocking()
            now = time.time()

            if key is not None:
                key = key.lower()
                if key == "q":
                    break
                if key == "i":
                    active_x, active_yaw = args.speed, 0.0
                    active_cmd_ts = now
                elif key == "k":
                    active_x, active_yaw = -args.speed, 0.0
                    active_cmd_ts = now
                elif key == "j":
                    active_x, active_yaw = 0.0, yaw_rate
                    active_cmd_ts = now
                elif key == "l":
                    active_x, active_yaw = 0.0, -yaw_rate
                    active_cmd_ts = now
                elif key == " ":
                    active_x, active_yaw = 0.0, 0.0
                    active_cmd_ts = now
                elif key == "h":
                    controller.go_home()

            if now - active_cmd_ts > args.repeat_timeout:
                active_x, active_yaw = 0.0, 0.0

            controller.send_velocity(active_x, 0.0, active_yaw, record=False)
            time.sleep(dt)
    finally:
        controller.stop()
    print("IJKL movement test ended.")


if __name__ == "__main__":
    main()
