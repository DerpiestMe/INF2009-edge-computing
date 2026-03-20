import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2


class ZoneManager:
    """Defines restricted regions and checks whether a person stands inside them."""

    def __init__(self, zones: Optional[List[Dict]] = None) -> None:
        self.zones = zones or [
            {
                "id": "zone_A",
                "name": "Restricted Area",
                "x1": 180,
                "y1": 100,
                "x2": 500,
                "y2": 430,
            }
        ]

    @staticmethod
    def get_bottom_center(bbox) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) // 2, y2

    @staticmethod
    def point_in_zone(point, zone) -> bool:
        px, py = point
        return zone["x1"] <= px <= zone["x2"] and zone["y1"] <= py <= zone["y2"]

    def assign_zone(self, detection: Dict) -> Optional[Dict]:
        point = self.get_bottom_center(detection["bbox"])
        for zone in self.zones:
            if self.point_in_zone(point, zone):
                return zone
        return None

    def annotate_detections_with_zone(self, detections: List[Dict]) -> List[Dict]:
        output: List[Dict] = []
        for det in detections:
            zone = self.assign_zone(det)
            enriched = det.copy()
            enriched["inside_zone"] = zone is not None
            enriched["zone_id"] = zone["id"] if zone else None
            enriched["zone_name"] = zone["name"] if zone else None
            output.append(enriched)
        return output

    def draw_zones(self, frame):
        for zone in self.zones:
            cv2.rectangle(frame, (zone["x1"], zone["y1"]), (zone["x2"], zone["y2"]), (255, 0, 0), 2)
            cv2.putText(
                frame,
                zone["name"],
                (zone["x1"], max(20, zone["y1"] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )
        return frame


class IntrusionEventManager:
    """Tracks intrusions in restricted zones with confirmation and cooldown."""

    def __init__(
        self,
        confirm_frames: int = 4,
        cooldown_seconds: float = 5.0,
        snapshot_dir: str = "snapshots",
    ) -> None:
        self.confirm_frames = confirm_frames
        self.cooldown_seconds = cooldown_seconds
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(exist_ok=True)

        self._intrusion_counter = 0
        self._last_event_ts = 0.0
        self._frame_history: deque = deque(maxlen=confirm_frames)

    def process(self, frame, detections: List[Dict]) -> Optional[Dict]:
        """
        Process a frame and detections to identify intrusion events.
        Returns an intrusion event dict if one is confirmed, otherwise None.
        """
        now = time.time()

        # Check if any person is inside a restricted zone
        has_intrusion = any(det.get("inside_zone", False) for det in detections)

        if has_intrusion:
            self._intrusion_counter += 1
            self._frame_history.append(True)
        else:
            self._intrusion_counter = 0
            self._frame_history.append(False)

        # Confirm intrusion after N consecutive frames with detection
        if self._intrusion_counter >= self.confirm_frames:
            # Check cooldown: prevent duplicate events
            if now - self._last_event_ts >= self.cooldown_seconds:
                self._last_event_ts = now
                self._intrusion_counter = 0

                # Save snapshot
                snapshot_path = self.snapshot_dir / f"intrusion_{int(now)}_{self._intrusion_counter}.jpg"
                cv2.imwrite(str(snapshot_path), frame)

                intrusion_in_zone = [det for det in detections if det.get("inside_zone")]
                event = {
                    "timestamp": now,
                    "type": "intrusion",
                    "zone_id": intrusion_in_zone[0].get("zone_id") if intrusion_in_zone else None,
                    "num_persons": len(intrusion_in_zone),
                    "snapshot_path": str(snapshot_path),
                    "confirm_frames": self.confirm_frames,
                }
                return event

        return None