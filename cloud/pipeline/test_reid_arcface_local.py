import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def _load_arcface():
    try:
        from insightface.app import FaceAnalysis  # type: ignore
        return FaceAnalysis
    except Exception as exc:
        print(f"[ERROR] insightface is not available: {exc}")
        print("Install with: pip install insightface onnxruntime")
        return None


class ArcFaceReID:
    def __init__(self, whitelist_dir: Path, threshold: float, det_size=(640, 640)):
        self.whitelist_dir = Path(whitelist_dir)
        self.threshold = float(threshold)
        self.det_size = det_size
        self._embeddings = []
        self._init_model()
        self._load_whitelist()

    def _init_model(self):
        FaceAnalysis = _load_arcface()
        if FaceAnalysis is None:
            sys.exit(1)
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=self.det_size)

    def _get_faces(self, img_bgr):
        return self.app.get(img_bgr)

    def _embed_first(self, img_bgr):
        faces = self._get_faces(img_bgr)
        if not faces:
            return None, []
        face = faces[0]
        emb = face.normed_embedding.astype(np.float32)
        boxes = [face.bbox.astype(int).tolist()]  # [x1,y1,x2,y2]
        return emb, boxes

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
            emb, _boxes = self._embed_first(img)
            if emb is None:
                print(f"[WARN] No face found in whitelist image: {path.name}")
                continue
            self._embeddings.append((path.stem, emb))
        print(f"[INFO] Whitelist loaded: {len(self._embeddings)} identities")

    def match(self, img_bgr, top_k: int = 3):
        if not self._embeddings:
            return {"matched": False, "name": "Unknown", "score": 0.0, "face_locations": []}

        emb, boxes = self._embed_first(img_bgr)
        if emb is None:
            return {"matched": False, "name": "Unknown", "score": 0.0, "face_locations": boxes}

        ranked = sorted(
            [(name, float(np.dot(emb, known))) for name, known in self._embeddings],
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]
        best_name, best_score = ranked[0] if ranked else ("Unknown", 0.0)
        matched = best_score >= self.threshold
        return {
            "matched": matched,
            "name": best_name if matched else "Unknown",
            "score": best_score,
            "face_locations": boxes,
            "top_matches": ranked,
        }


def main():
    parser = argparse.ArgumentParser(description="Local ArcFace re-identification test runner")
    parser.add_argument("--input-dir", default="cloud/test_snapshots", help="Folder of snapshots to test")
    parser.add_argument("--whitelist-dir", default=os.getenv("WHITELIST_DIR", "cloud/whitelist"))
    parser.add_argument("--threshold", type=float, default=float(os.getenv("REID_THRESHOLD_ARCFACE", "0.55")))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="Output JSON per image")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"[ERROR] Input dir not found: {input_dir}")
        sys.exit(1)

    reid = ArcFaceReID(Path(args.whitelist_dir), threshold=args.threshold)

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
