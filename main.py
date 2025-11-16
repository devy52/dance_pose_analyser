# main.py
from app.processor import PoseProcessor
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
model = BASE_DIR / "model" / "pose_landmarker_heavy.task"

def main():
    input_video = "C:\\Users\\hp\\Downloads\\Dance1.mp4"
    output_video = "C:\\Users\\hp\\Downloads\\landmarked_Dance1.mp4"
    model_path = model

    print("Starting pose processing...")

    processor = PoseProcessor(model_path=str(model_path), mode="VIDEO")

    success = processor.process_video(
        input_path=input_video,
        output_path=output_video
    )

    if success:
        print(f"✔ Processing complete. Output saved to {output_video}")
    else:
        print("❌ Processing failed.")

if __name__ == "__main__":
    main()
