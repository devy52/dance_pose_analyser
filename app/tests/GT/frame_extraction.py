import os
import shutil
import cv2

def save_frames_from_video(video_path, output_dir="tests/data", num_frames=3):
    # Create/Reset output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)   # delete folder and contents
    os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video:", video_path)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames in video:", total_frames)

    # Choose evenly spaced frames
    frame_indices = [
        int(total_frames * (i+1) / (num_frames + 1))
        for i in range(num_frames)
    ]

    print("Saving frames at indices:", frame_indices)

    saved = 0
    current_index = 0

    while saved < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if current_index in frame_indices:
            filename = os.path.join(output_dir, f"frame{saved+1}.jpg")
            cv2.imwrite(filename, frame)
            print("Saved:", filename)
            saved += 1

        current_index += 1

    cap.release()
    print("✔ Done. Frames saved in", output_dir)


# -----------------------------
# Run the function
# -----------------------------
if __name__ == "__main__":
    video_path = r"C:\Users\hp\Downloads\Dance1.mp4"  # change if needed
    save_frames_from_video(video_path, num_frames=3)
