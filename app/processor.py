# app/processor.py
import cv2
import mediapipe as mp
import numpy as np
import subprocess
from pathlib import Path
from typing import Union
from .logger import logger

POSE_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,7),
    (0,4), (4,5), (5,6), (6,8),
    (9,10),
    (11,12),
    (11,13), (13,15), (15,17), (15,19), (15,21),
    (12,14), (14,16), (16,18), (16,20), (16,22),
    (11,23), (12,24),
    (23,24),
    (23,25), (25,27), (27,29), (29,31),
    (24,26), (26,28), (28,30), (30,32)
]

def draw_landmarks_on_image(rgb_image, detection_result):
    annotated = rgb_image.copy()

    if not detection_result.pose_landmarks:
        return annotated

    h, w, _ = annotated.shape

    for landmarks in detection_result.pose_landmarks:
        for lm in landmarks:
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            cv2.circle(annotated, (cx, cy), 3, (0, 255, 0), -1)

        for a, b in POSE_CONNECTIONS:
            pa, pb = landmarks[a], landmarks[b]
            ax, ay = int(pa.x * w), int(pa.y * h)
            bx, by = int(pb.x * w), int(pb.y * h)
            cv2.line(annotated, (ax, ay), (bx, by), (0, 255, 0), 2)

    return annotated

def make_browser_friendly_mp4(input_path: Union[str, Path]) -> str:
    """
    Converts raw OpenCV MP4 into browser-playable H.264 MP4 with faststart enabled.
    """
    input_path = Path(input_path)
    output_path = input_path.with_name(input_path.stem + "_fixed.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-movflags", "faststart",
        "-acodec", "aac",
        "-b:a", "128k",
        str(output_path)
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return str(output_path)
    return str(output_path)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
model = BASE_DIR / "model" / "pose_landmarker_heavy.task"

class PoseProcessor:
    def __init__(self, model_path=str(model), mode="VIDEO"):
        logger.info(f"Initializing PoseLandmarker in {mode} mode...")

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        run_mode = (
            VisionRunningMode.IMAGE if mode.upper() == "IMAGE"
            else VisionRunningMode.VIDEO
        )

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=run_mode,
            output_segmentation_masks=True
        )

        self.landmarker = PoseLandmarker.create_from_options(options)
        self.mode = mode.upper()
        logger.info(f"PoseLandmarker initialized in {self.mode} mode.")

    def process_video(self, input_path: str, output_path: str):
        logger.info(f"Processing video: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error("Could not open video.")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Temporary raw output (OpenCV)
        raw_output = output_path.replace(".mp4", "_raw.mp4")

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(raw_output, fourcc, fps, (width, height))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Total frames: {frame_count}, FPS: {fps}")

        timestamp = 0  
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = self.landmarker.detect_for_video(mp_image, timestamp)
            annotated = draw_landmarks_on_image(rgb, result)

            out.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            timestamp += int(1000 / fps)
            idx += 1

        cap.release()
        out.release()

        logger.info(f"Raw OpenCV output saved to: {raw_output}")

        # ------------------------------
        # Final re-encode to browser-safe MP4
        # ------------------------------
        fixed_output = output_path

        cmd = [
            "ffmpeg", "-y",
            "-i", raw_output,
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "medium",
            "-movflags", "faststart",
            fixed_output
        ]

        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        logger.info(f"Browser-compatible MP4 saved to: {fixed_output}")

        # Remove raw OpenCV file
        try:
            Path(raw_output).unlink()
        except:
            pass

        return True

