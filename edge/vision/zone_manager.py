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