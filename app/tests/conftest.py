# tests/conftest.py
import pytest
import logging
import sys
import os
import json
import cv2

# --- Ensure "app" can be imported ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.processor import PoseProcessor



# --- Configure test logger ---
logger = logging.getLogger("test-logger")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("app/logs/test.log", mode="w")
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.info("=== Starting Test Suite ===")

@pytest.fixture(scope="session")
def pose_processor():
    logger.info("Initializing PoseProcessor for tests (IMAGE mode)...")
    processor = PoseProcessor(model_path="app/model/pose_landmarker_heavy.task", mode="IMAGE")
    return processor

@pytest.fixture(scope="session")
def test_frame():
    path = "app/tests/data/frame1.jpg"
    logger.info(f"Loading test frame from {path}")
    frame = cv2.imread(path)
    assert frame is not None, "Test frame not found!"
    return frame

def get_gt_files():
    data_dir = "app/tests/data"
    return [
        os.path.join(data_dir, f)
        for f in sorted(os.listdir(data_dir))
        if f.startswith("frame") and f.endswith(".json")
    ]

@pytest.fixture(params=get_gt_files(), scope="session")
def ground_truth(request):
    """Provide each GT file as a separate test case."""
    path = request.param
    logger.info(f"Loading GT from {path}")
    with open(path, "r") as f:
        return json.load(f)
