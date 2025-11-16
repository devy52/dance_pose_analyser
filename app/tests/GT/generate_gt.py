# tests/GT/generate_gt.py
import sys
import os
import json
import cv2
import mediapipe as mp

# ----- Add project root so "from app.processor import PoseProcessor" works -----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.processor import PoseProcessor
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
model = BASE_DIR / "model" / "pose_landmarker_heavy.task"


def generate_gt_for_frames(
    data_dir="app/tests/data",
    
    model_path=model,
    assume_fps: float | None = None,
):
    """
    Generate GT JSON files for all frame*.jpg files found in data_dir.

    If the landmarker is in VIDEO running mode, this function uses detect_for_video(...)
    with timestamps. If it's in IMAGE mode, it uses detect(...).

    `assume_fps` can be given (e.g. 30.0). If None, timestamps use 33 ms per frame (≈30 FPS).
    """
    processor = PoseProcessor(model_path=str(model_path), mode="VIDEO")

    # find frames
    frames = sorted(f for f in os.listdir(data_dir) if f.startswith("frame") and f.endswith(".jpg"))
    if not frames:
        print("❌ No frames found in", data_dir)
        return

    print("Frames found:", frames)

    # choose timestamp increment (milliseconds)
    if assume_fps and assume_fps > 0:
        ms_per_frame = int(1000.0 / assume_fps)
    else:
        ms_per_frame = 33  # default ~30 FPS

    # detect API choice
    use_detect_for_video = hasattr(processor.landmarker, "detect_for_video")
    print("Landmarker API:", "detect_for_video (VIDEO mode)" if use_detect_for_video else "detect (IMAGE mode)")

    for idx, frame_file in enumerate(frames):
        frame_path = os.path.join(data_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print("❌ Could not read:", frame_path)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # call appropriate API
        try:
            if use_detect_for_video:
                timestamp_ms = idx * ms_per_frame
                result = processor.landmarker.detect_for_video(mp_image, timestamp_ms)
            else:
                result = processor.landmarker.detect(mp_image)
        except Exception as e:
            print(f"❌ Detection failed for {frame_file}: {e}")
            continue

        if not getattr(result, "pose_landmarks", None):
            print(f"⚠ No landmarks detected for {frame_file}, skipping.")
            continue

        landmarks = result.pose_landmarks[0]
        out = [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in landmarks]

        gt_path = os.path.join(data_dir, f"{frame_file.replace('.jpg','')}_gt.json")
        with open(gt_path, "w") as f:
            json.dump(out, f, indent=4)

        print(f"✔ Saved GT: {gt_path}")


if __name__ == "__main__":
    # If your frames come from a video at known fps, pass assume_fps=59.94 or 30 etc.
    # Example: generate_gt_for_frames(assume_fps=59.94)
    generate_gt_for_frames()
