import argparse
import logging
import signal
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puppypi_movement import PuppyPiMovementController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basic PuppyPi forward movement test")
    parser.add_argument("--speed", type=float, default=8.0, help="Forward speed in cm/s")
    parser.add_argument("--duration", type=float, default=1.5, help="Forward movement duration in seconds")
    parser.add_argument("--settle", type=float, default=0.5, help="Pause before/after motion in seconds")
    parser.add_argument("--go-home", action="store_true", help="Call go_home at end")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    controller = PuppyPiMovementController()

    if not controller.start():
        print("[ERROR] Movement controller failed to initialize (ROS stack unavailable).")
        return

    running = True

    def _shutdown(_signum: int, _frame) -> None:
        nonlocal running
        running = False
        controller.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print("[INFO] Forward movement test starting...")
    print(f"[INFO] Command: x={args.speed:.2f} cm/s for {args.duration:.2f}s")
    try:
        time.sleep(max(0.0, args.settle))
        if running:
            controller.send_velocity(x=args.speed, y=0.0, yaw_rate=0.0, record=False)
            time.sleep(max(0.0, args.duration))
        controller.stop()
        time.sleep(max(0.0, args.settle))
        if args.go_home and running:
            controller.go_home()
    finally:
        controller.stop()
    print("[INFO] Forward movement test complete.")


if __name__ == "__main__":
    main()
