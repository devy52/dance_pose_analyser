from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from contextlib import asynccontextmanager
from pathlib import Path
from app.processor import PoseProcessor
import shutil
import uuid

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = BASE_DIR / "model" / "pose_landmarker_heavy.task"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

import subprocess

def convert_to_h264(input_file: str) -> str:
    converted = input_file.replace(".mp4", "_h264.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-i", input_file,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-movflags", "faststart",
        converted
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return converted


# ------------------------------
# Modern lifespan event handler
# ------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 FastAPI starting up...")

    yield  # <-- the app runs here

    print("🧹 Cleaning up files...")
    for directory in [UPLOAD_DIR, OUTPUT_DIR]:
        for file in directory.glob("*"):
            try:
                file.unlink()
            except:
                pass
    print("✔ Cleanup complete")


# Initialize FastAPI
app = FastAPI(title="Dance Pose Analyzer", lifespan=lifespan)

# Static + templates
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

processor = PoseProcessor(str(MODEL_PATH))


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(400, "Invalid video type")

    uid = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{uid}_{file.filename}"
    output_name = f"{uid}_annotated.mp4"
    output_path = OUTPUT_DIR / output_name

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Convert input to H.264 (browser + OpenCV safe)
    h264_input_path = convert_to_h264(str(input_path))

    success = processor.process_video(h264_input_path, str(output_path))

    if not success:
        raise HTTPException(500, "Processing failed")

    return JSONResponse({"output": output_name})


@app.get("/preview/{file_name}")
def preview(file_name: str):
    file_path = OUTPUT_DIR / file_name
    if not file_path.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(str(file_path))


@app.get("/download/{file_name}")
def download(file_name: str):
    file_path = OUTPUT_DIR / file_name
    if not file_path.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(str(file_path), media_type="video/mp4", filename=file_name)


# Error Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
