import os
import sys
from pathlib import Path
import argparse
import json

import numpy as np
import cv2


def load_face_backend():
    try:
        import face_recognition  # type: ignore
        return face_recognition, True
    except Exception:
        return None, False


class WhitelistReID:
    def __init__(self, whitelist_dir: Path, threshold: float, margin: float):
        self.whitelist_dir = Path(whitelist_dir)
        self.threshold = float(threshold)
        self.margin = float(margin)
        self._face_recognition, self._use_face_recognition = load_face_backend()
        self._embeddings = []
        self._load_whitelist()

    def _embed(self, img_bgr):
        if self._use_face_recognition and self._face_recognition:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            boxes = self._face_recognition.face_locations(rgb)
            encodings = self._face_recognition.face_encodings(rgb, boxes)
            if not encodings:
                return None, []
            return np.array(encodings[0], dtype=np.float32), boxes

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (64, 64))
        vec = small.astype(np.float32).flatten()
        norm = np.linalg.norm(vec) + 1e-6
        return vec / norm, []

    def _load_whitelist(self):
        self._embeddings = []
        if not self.whitelist_dir.exists():
            print(f"[WARN] Whitelist directory missing: {self.whitelist_dir}")
            return
        for path in sorted(self.whitelist_dir.glob("*")):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = cv2.imread(str(path))
            if img is None:
                continue
            emb, _boxes = self._embed(img)
            if emb is None:
                print(f"[WARN] No face found in whitelist image: {path.name}")
                continue
            self._embeddings.append((path.stem, emb))
        print(f"[INFO] Whitelist loaded: {len(self._embeddings)} identities")

    def match(self, img_bgr, top_k: int = 3):
        if not self._embeddings:
            return {"matched": False, "name": "Unknown", "score": 0.0, "face_locations": []}

        emb, boxes = self._embed(img_bgr)
        if emb is None:
            return {"matched": False, "name": "Unknown", "score": 0.0, "face_locations": boxes}

        best_name = "Unknown"
        best_score = 0.0

        if self._use_face_recognition and self._face_recognition is not None:
            known = [e for _, e in self._embeddings]
            distances = self._face_recognition.face_distance(known, emb)
            if len(distances) == 0:
                return {"matched": False, "name": "Unknown", "score": 0.0, "face_locations": boxes}
            best_idx = int(np.argmin(distances))
            best_name = self._embeddings[best_idx][0]
            best_score = 1.0 - float(distances[best_idx])
            ranked = sorted(
                [(self._embeddings[i][0], float(1.0 - distances[i])) for i in range(len(distances))],
                key=lambda x: x[1],
                reverse=True,
            )[:top_k]
            margin_ok = True if self.margin <= 0 else False
            if self.margin > 0 and len(ranked) > 1:
                second_score = ranked[1][1]
                margin_ok = (best_score - second_score) >= self.margin
            matched = best_score >= self.threshold and margin_ok
            return {
                "matched": matched,
                "name": best_name if matched else "Unknown",
                "score": best_score,
                "face_locations": boxes,
                "top_matches": ranked,
                "margin_ok": margin_ok,
            }

        ranked = sorted(
            [(name, float(np.dot(emb, known))) for name, known in self._embeddings],
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]
        if ranked:
            best_name, best_score = ranked[0]
        margin_ok = True if self.margin <= 0 else False
        if self.margin > 0 and len(ranked) > 1:
            second_score = ranked[1][1]
            margin_ok = (best_score - second_score) >= self.margin
        matched = best_score >= self.threshold and margin_ok
        return {
            "matched": matched,
            "name": best_name if matched else "Unknown",
            "score": best_score,
            "face_locations": boxes,
            "top_matches": ranked,
            "margin_ok": margin_ok,
        }


def main():
    parser = argparse.ArgumentParser(description="Local re-identification test runner")
    parser.add_argument("--input-dir", default="cloud/test_snapshots", help="Folder of snapshots to test")
    parser.add_argument("--whitelist-dir", default=os.getenv("WHITELIST_DIR", "cloud/whitelist"))
    parser.add_argument("--threshold", type=float, default=float(os.getenv("REID_THRESHOLD", "0.65")))
    parser.add_argument("--margin", type=float, default=float(os.getenv("REID_MARGIN", "0.05")))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="Output JSON per image")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"[ERROR] Input dir not found: {input_dir}")
        sys.exit(1)

    reid = WhitelistReID(Path(args.whitelist_dir), threshold=args.threshold, margin=args.margin)

    images = sorted([p for p in input_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not images:
        print(f"[WARN] No images found in {input_dir}")
        return

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Could not read {img_path.name}")
            continue
        result = reid.match(img, top_k=args.top_k)
        result["file"] = img_path.name
        if args.json:
            print(json.dumps(result))
        else:
            print(
                f"{img_path.name}: matched={result['matched']} name={result['name']} "
                f"score={result['score']:.2f} faces={len(result.get('face_locations') or [])} "
                f"top={result.get('top_matches')}"
            )


if __name__ == "__main__":
    main()
