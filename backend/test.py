"""
Quick CLI test — no frontend needed.
Usage:
    python3 test.py path/to/image.jpg
    python3 test.py path/to/video.mp4
"""

import sys
import cv2
from model import Videocaption
from video_utils import extract_frames

#PROMPT = (
 #   "Describe the crash scene. Focus on vehicles involved, "
  #  "damage visible, road conditions, passenger condition, and any hazards."
#)
PROMPT = (
  # "Describe the scene. Is anyone injured? Are there kids present? "
   "Are they wearing seatbelt? "
  # "Is there any blood? "
)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test.py <image_or_video_path>")
        sys.exit(1)

    path = sys.argv[1]

    print("[INFO] Loading model (downloads ~1 GB on first run)...")
    model = Videocaption()

    if path.lower().endswith(("mp4", "avi", "mov")):
        print("[INFO] Extracting frames from video...")
        frames = extract_frames(path)
    else:
        img = cv2.imread(path)
        if img is None:
            print(f"[ERROR] Could not read file: {path}")
            sys.exit(1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames = [img]

    print("[INFO] Generating caption...")
    result = model.generate_caption(frames, PROMPT)
    print("\n--- CAPTION ---")
    print(result)

if __name__ == "__main__":
    main()
