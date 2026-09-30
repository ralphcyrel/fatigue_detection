"""
Eye-landmark check - a standing diagnostic for new drivers.

Shows where dlib's 12 eye landmarks sit on a driver's face and measures the
closed/open EAR ratio directly - the same contrast test enrollment applies
(``modules.calibration.EAR_CONTRAST_MAX``), with the same closed-eye timing
(``ClosedEyeCapture``). Use it before enrolling a driver whose EAR looks
unusual, or when an enrollment fails its closed-eye check, to see whether
the landmarks follow the eyelid.

Run on the unit, from the repository root, with the driver in the seat::

    python tools/eye_check.py --label driver8
    python tools/eye_check.py --label driver8 --no-preview   # headless

Nothing is sent to the backend and no GPIO is touched (the relay is left
alone); the operator tells the driver when to close and open their eyes (the
terminal and the preview both show the cue).

Writes ``logs/eye_checks/<label>_<UTC stamp>/``:

* ``open.png`` / ``closed.png`` - full frame, all 68 points, eye points
  highlighted, EAR printed;
* ``open_eyes.png`` / ``closed_eyes.png`` / ``compare.png`` - both eyes
  enlarged, the six points per eye numbered (dlib 36-47), the two vertical
  distances EAR measures in green and the width in blue;
* ``series.csv`` - per frame: phase, time, EAR (both eyes, left, right) and
  the 12 eye-landmark coordinates;
* ``summary.json`` - open median, closed median, ratio per eye and averaged,
  and the verdict against the enrollment rules.

What to look for in the images: the upper-lid points (37, 38 and 43, 44)
should sit on the lash line, not on a skin fold or the brow, and on the
closed image they should come down to meet the lower-lid points (41, 40 and
47, 46). A ratio at or below ``EAR_CONTRAST_MAX`` means the pipeline will see
this driver's eyes close.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config  # noqa: E402
from main import Camera  # noqa: E402  (same capture settings as the pipeline)
from modules.calibration import (EAR_CONTRAST_MAX, EAR_NO_CLOSURE_RATIO,  # noqa: E402
                                 ClosedEyeCapture, ear_calibration_problem)
from modules.ear import EARCalculator, LEFT_EYE, RIGHT_EYE  # noqa: E402
from modules.landmark import LandmarkExtractor  # noqa: E402

OUT_DIR = config.LOGS_DIR / "eye_checks"
# Countdown before each phase, so the driver is settled when capture starts.
LEAD_S = 3.0
# Keep every Nth face frame's image for picking the snapshot (memory bound).
KEEP_EVERY = 3
# Enlargement of the eye crop.
ZOOM = 4
EYES = (("right", RIGHT_EYE), ("left", LEFT_EYE))   # subject's right / left


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--label", required=True, help="e.g. driver8 (used in the folder name)")
    p.add_argument("--open-s", type=float, default=5.0, help="seconds of open-eye capture")
    p.add_argument("--no-preview", action="store_true", help="no window (headless)")
    p.add_argument("--out", type=Path, default=OUT_DIR, help=f"default {OUT_DIR}")
    return p.parse_args()


class Session:
    """Camera + landmarks + optional preview."""

    def __init__(self, preview: bool) -> None:
        self.camera = Camera()
        self.extractor = LandmarkExtractor(str(config.LANDMARK_MODEL), config.SCALE_FACTOR)
        self.ear = EARCalculator()
        self.preview = preview

    def frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Next ``(frame, landmarks)``; landmarks ``None`` without a face."""
        frame = self.camera.read()
        if frame is None:
            return None, None
        landmarks, _ = self.extractor.extract(frame)
        return frame, landmarks

    def show(self, frame: np.ndarray, text: str) -> None:
        if not self.preview:
            return
        shown = frame.copy()
        cv2.rectangle(shown, (0, 0), (shown.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(shown, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                    cv2.LINE_AA)
        try:
            cv2.imshow("eye check", shown)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt
        except cv2.error:
            self.preview = False   # no display after all

    def countdown(self, text: str) -> None:
        print(f"\n>>> {text}")
        start = time.time()
        announced = None
        while (left := LEAD_S - (time.time() - start)) > 0:
            if announced != int(left) + 1:
                announced = int(left) + 1
                print(f"    {announced}...", flush=True)
            frame, _ = self.frame()
            if frame is not None:
                self.show(frame, f"{text}  {left:.0f}")

    def close(self) -> None:
        self.camera.release()
        if self.preview:
            cv2.destroyAllWindows()


def eye_ears(calc: EARCalculator, landmarks: np.ndarray) -> Tuple[float, float, float]:
    """(average, right, left) EAR, exactly as the pipeline computes the average."""
    right = calc.compute_ear(landmarks[RIGHT_EYE[0]:RIGHT_EYE[1]])
    left = calc.compute_ear(landmarks[LEFT_EYE[0]:LEFT_EYE[1]])
    return calc.compute_average_ear(landmarks), right, left


def draw_full(frame: np.ndarray, landmarks: np.ndarray, title: str) -> np.ndarray:
    out = frame.copy()
    eye_idx = set(range(*RIGHT_EYE)) | set(range(*LEFT_EYE))
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(out, (int(round(x)), int(round(y))), 2 if i in eye_idx else 1,
                   (0, 255, 255) if i in eye_idx else (255, 255, 255), -1)
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(out, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
                cv2.LINE_AA)
    return out


def draw_zoom(frame: np.ndarray, landmarks: np.ndarray, title: str) -> np.ndarray:
    """Both eyes enlarged, with the EAR geometry drawn on sub-pixel landmark positions."""
    pts = landmarks[RIGHT_EYE[0]:LEFT_EYE[1]]
    span = pts[:, 0].max() - pts[:, 0].min()
    pad = 0.15 * span
    x0 = int(max(pts[:, 0].min() - pad, 0))
    x1 = int(min(pts[:, 0].max() + pad, frame.shape[1]))
    y0 = int(max(pts[:, 1].min() - 2 * pad, 0))
    y1 = int(min(pts[:, 1].max() + 2 * pad, frame.shape[0]))
    crop = cv2.resize(frame[y0:y1, x0:x1], None, fx=ZOOM, fy=ZOOM,
                      interpolation=cv2.INTER_CUBIC)

    def at(i: int) -> Tuple[int, int]:
        x, y = landmarks[i]
        return int(round((x - x0) * ZOOM)), int(round((y - y0) * ZOOM))

    for _, (start, stop) in EYES:
        p = [at(i) for i in range(start, stop)]   # p1..p6 as in ear.py
        cv2.polylines(crop, [np.array(p, np.int32)], True, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(crop, p[0], p[3], (255, 128, 0), 1, cv2.LINE_AA)   # width
        cv2.line(crop, p[1], p[5], (0, 255, 0), 2, cv2.LINE_AA)     # vertical 1
        cv2.line(crop, p[2], p[4], (0, 255, 0), 2, cv2.LINE_AA)     # vertical 2
        for k, (x, y) in enumerate(p):
            cv2.circle(crop, (x, y), 3, (0, 0, 255), -1)
            # Lower-lid labels (p5, p6) go below the point so a closed eye
            # does not stack them on the upper-lid ones.
            cv2.putText(crop, str(start + k), (x + 4, y + 16 if k in (4, 5) else y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    bar = np.zeros((30, crop.shape[1], 3), np.uint8)
    cv2.putText(bar, title, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1,
                cv2.LINE_AA)
    return np.vstack([bar, crop])


def side_by_side(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    h = max(a.shape[0], b.shape[0])
    pad = lambda im: np.vstack([im, np.zeros((h - im.shape[0], im.shape[1], 3), np.uint8)])  # noqa: E731
    return np.hstack([pad(a), np.zeros((h, 8, 3), np.uint8), pad(b)])


def main() -> int:
    args = parse_args()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = args.out / f"{args.label}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    s = Session(preview=not args.no_preview)
    rows: List[Dict[str, Any]] = []
    kept: Dict[str, List[Tuple[float, np.ndarray, np.ndarray]]] = {"open": [], "closed": []}

    def record(phase: str, t: float, frame: np.ndarray, landmarks: np.ndarray, n: int) -> float:
        ear, right, left = eye_ears(s.ear, landmarks)
        row: Dict[str, Any] = {"phase": phase, "t": round(t, 3), "ear": round(ear, 4),
                               "ear_right": round(right, 4), "ear_left": round(left, 4)}
        for i in range(RIGHT_EYE[0], LEFT_EYE[1]):
            row[f"x{i}"], row[f"y{i}"] = round(float(landmarks[i][0]), 2), round(float(landmarks[i][1]), 2)
        rows.append(row)
        if n % KEEP_EVERY == 0:
            kept[phase].append((ear, frame.copy(), landmarks.copy()))
        return ear

    try:
        # ---- open ---------------------------------------------------------
        s.countdown("EYES OPEN - look at the camera normally")
        print(f"    capturing {args.open_s:.0f}s eyes open...")
        start, n = time.time(), 0
        while time.time() - start < args.open_s:
            frame, lm = s.frame()
            if frame is None:
                continue
            if lm is None:
                s.show(frame, "NO FACE")
                continue
            ear = record("open", time.time() - start, frame, lm, n)
            n += 1
            s.show(frame, f"EYES OPEN  EAR {ear:.3f}")

        # ---- closed (enrollment timing) -----------------------------------
        s.countdown("CLOSE YOUR EYES at zero and keep them shut until told to open")
        cap = ClosedEyeCapture()
        cap.start(time.time())
        print(f"\a    CLOSE NOW - capturing {cap.duration_s:.0f}s "
              f"(first {cap.settle_s:.1f}s discarded)", flush=True)
        n = 0
        while not cap.done(time.time()):
            frame, lm = s.frame()
            if frame is None:
                continue
            if lm is None:
                s.show(frame, "NO FACE - keep facing the camera")
                continue
            now = time.time()
            ear = eye_ears(s.ear, lm)[0]
            cap.update(ear, now)
            if cap.elapsed(now) >= cap.settle_s:   # snapshots from the measured part only
                record("closed", cap.elapsed(now), frame, lm, n)
                n += 1
            else:
                rows.append({"phase": "closed_settle", "t": round(cap.elapsed(now), 3),
                             "ear": round(ear, 4)})
            s.show(frame, f"EYES CLOSED  EAR {ear:.3f}")
        print("\a    OPEN YOUR EYES", flush=True)
    finally:
        s.close()

    # ---- results --------------------------------------------------------------
    def med(phase: str, key: str) -> Optional[float]:
        vals = [r[key] for r in rows if r["phase"] == phase]
        return float(np.median(vals)) if vals else None

    summary: Dict[str, Any] = {"label": args.label, "captured_at_utc": stamp,
                               "device_id": config.DEVICE_ID,
                               "frames_open": sum(r["phase"] == "open" for r in rows),
                               "frames_closed": sum(r["phase"] == "closed" for r in rows)}
    for key in ("ear", "ear_right", "ear_left"):
        o, c = med("open", key), med("closed", key)
        summary[key] = {"open_median": o, "closed_median": c,
                        "ratio": None if not o or c is None else round(c / o, 4)}
    ratio = summary["ear"]["ratio"]
    open_med, closed_med = summary["ear"]["open_median"], summary["ear"]["closed_median"]
    if ratio is None:
        verdict = "NO RESULT - too few face frames in one of the phases"
    else:
        problem = ear_calibration_problem(open_med, closed_med)
        if problem is None:
            verdict = f"PASS - ratio {ratio:.2f} <= {EAR_CONTRAST_MAX:.2f}: the landmarks follow the eyelid"
        elif ratio >= EAR_NO_CLOSURE_RATIO:
            verdict = (f"FAIL - ratio {ratio:.2f}: no eyelid movement (eyes not closed, or "
                       f"landmarks not tracking the lids - check the images)")
        else:
            verdict = f"FAIL - {problem}"
    summary["verdict"] = verdict
    summary["contrast_max"] = EAR_CONTRAST_MAX
    summary["note"] = ("open value is a median over the open phase; enrollment's ear_baseline "
                       "is a 60 s mean including blinks, typically 0-4 % lower")

    fields = ["phase", "t", "ear", "ear_right", "ear_left"] + [
        f"{a}{i}" for i in range(RIGHT_EYE[0], LEFT_EYE[1]) for a in ("x", "y")]
    with (out / "series.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    zooms = {}
    for phase, target in (("open", open_med), ("closed", closed_med)):
        if not kept[phase] or target is None:
            continue
        ear, frame, lm = min(kept[phase], key=lambda k: abs(k[0] - target))
        title = f"{args.label} {phase.upper()}  EAR {ear:.3f} (phase median {target:.3f})"
        cv2.imwrite(str(out / f"{phase}.png"), draw_full(frame, lm, title))
        zooms[phase] = draw_zoom(frame, lm, title)
        cv2.imwrite(str(out / f"{phase}_eyes.png"), zooms[phase])
    if len(zooms) == 2:
        cv2.imwrite(str(out / "compare.png"), side_by_side(zooms["open"], zooms["closed"]))

    print(f"\nOpen EAR   median {open_med}  (right {summary['ear_right']['open_median']}, "
          f"left {summary['ear_left']['open_median']})")
    print(f"Closed EAR median {closed_med}  (right {summary['ear_right']['closed_median']}, "
          f"left {summary['ear_left']['closed_median']})")
    print(f"Ratio {ratio}  (right {summary['ear_right']['ratio']}, left {summary['ear_left']['ratio']})")
    print(verdict)
    print(f"Written to {out}")
    return 0 if verdict.startswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
