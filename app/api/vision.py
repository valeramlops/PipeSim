from fastapi import APIRouter, UploadFile, File, HTTPException, Response, BackgroundTasks
import shutil
import uuid
from pathlib import Path
from ultralytics import YOLO
import cv2
from typing import List
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db, async_session
from app.models import DetectionRecord
from app.core.tasks import process_video_task
from app.core.celery_app import celery_app
from celery.result import AsyncResult

router = APIRouter()

# Create folder path
UPLOAD_DIR = Path("uploads/images")
UPLOAD_VIDEO_DIR = Path("uploads/videos")

# Automation create folder if not exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO("yolo11n.pt")

logger = logging.getLogger("uvicorn.error")

# Work logic with video (OpenCV)
def frame_generator(video_path: str):
    """
    Lazy iterator. Reads a video frame by frame without loading the entire video into RAM
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Failed to open videofile: {video_path}")
    
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read() # Read only one file
            if not ret:
                break # ret == False means that the video has ended

            yield frame_idx, frame # Send frame and "freeze" function
            frame_idx += 1
    finally:
        cap.release() # Make sure to release the resource (close the pipe)

# Function to save record in database
async def save_to_db_background(records: list):
    """
    Accepts an array of dictionaries and writes them to the database in one transaction
    """
    logger.info(f"[BACKGROUND] Start saving batch. Records count: {len(records)}")
    try:
        async with async_session() as db:
            db_records = [
                DetectionRecord(
                    id=rec["id"],
                    filename=rec["filename"],
                    result_json=rec["detections"]
                ) for rec in records
            ]
            db.add_all(db_records)
            await db.commit()
            logger.info("[BACKGROUND] Batch successfully saved in PostgreSQL!")
    except Exception as e:
        logger.error(f"[BACKGROUND CRASH] Critical error: {e}")

@router.post("/upload")
async def upload_image(
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File(...)
    ):
    """
    Endpoint for uploading images and real-time detection using YOLOv11
    """
    # 1. Security: check file format
    valid_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

    saved_files_info = []
    db_records_to_save = []
    response_data = []

    # 2. Generating unique filename and safe on disk
    for file in files:
        if not (file.content_type.startswith("image/")
                or file.filename.lower().endswith(valid_extensions)):
            continue

        detection_id = str(uuid.uuid4())
        unique_filename = f"{detection_id}_{file.filename}"
        file_path = UPLOAD_DIR / unique_filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        saved_files_info.append({
            "id": detection_id,
            "filename": file.filename,
            "path": str(file_path)
        })

    if not saved_files_info:
        raise HTTPException(
            status_code=400,
            detail="No valid images uploaded in the batch"
        )

    # YOLO Batch Inference
    path_for_yolo = [item["path"] for item in saved_files_info]

    # 3. Run through the AI
    results = model(path_for_yolo, conf=0.25)

    # Getting results
    for i, result in enumerate(results):
        real_detections = []
        for box in result.boxes:
            # Get data from PyTorch tensors
            coords = box.xyxy[0].tolist() # [x_min, y_min, x_max, y_max]
            conf = float(box.conf[0]) # Confidence
            class_id = int(box.cls[0]) # Internal class ID (for example: 0)
            class_name = model.names[class_id] # Converting id into text

            real_detections.append({
                "class": class_name,
                "confidence": round(conf, 2),
                "bbox": [round(c, 1) for c in coords]
            })
        file_info = saved_files_info[i]

        # Prepare data for json-answer to user
        response_data.append({
            "detection_id": file_info["id"],
            "original_filename": file_info["filename"],
            "detections": real_detections
        })

        # Filling the array for background recording in the database
        db_records_to_save.append({
            "id": file_info["id"],
            "filename": file_info["filename"],
            "detections": real_detections 
        })

    # 4. Send the entire batch to the database with a single background task
    background_tasks.add_task(
        save_to_db_background,
        records=db_records_to_save
    )
    
    return {
        "status": "success",
        "processed_count": len(response_data),
        "results": response_data,
        "message": f"Successfully processed {len(response_data)} images. DB save running in background"
    }

@router.post("/upload_video")
async def upload_video_endpoint(
    file: UploadFile = File(...)
):
    """
    An endpoint for video download. While just testing frame slicing 
    """
    valid_extrensions = (".mp4", ".avi", ".mov")
    if not file.filename.lower().endswith(valid_extrensions):
        raise HTTPException(
            status_code=400,
            detail="Supporting only .mp4, .avi, .mov"
        )
    
    # 1. Security saving video on hard disk
    video_id = str(uuid.uuid4())
    unique_filename = f"{video_id}_{file.filename}"
    video_path = UPLOAD_VIDEO_DIR / unique_filename

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.info(f"Video {unique_filename} successfully saved on disk")

    # 1.1 Defining output path for the worker
    output_filename = f"Processed_{unique_filename}"
    output_path = UPLOAD_VIDEO_DIR / output_filename

    # 2. Send task to Celery
    # .delay() method put task into Redis and instantly return control
    task = process_video_task.delay(str(video_path), str(output_path))

    # 3. Instantly responding to the user
    return {
        "status": "processing",
        "video_id": video_id,
        "task_id": task.id,
        "message": "Video successfully saved. Processing started in the background"
    }

@router.post("/debug-draw")
async def debug_draw_image(image: UploadFile = File(...)):
    """
    Return image with drawed borders
    """
    valid_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    is_image_mime = image.content_type.startswith("image/")
    is_valid_ext = image.filename.lower().endswith(valid_extensions)
    if not (is_image_mime or is_valid_ext):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. MIME: {image.content_type}"
        )
    
    # 1. Safe file on disk
    unique_filename = f"{uuid.uuid4()}_{image.filename}"
    file_path = UPLOAD_DIR / unique_filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # 2. Run through AI
    results = model(str(file_path), conf=0.25)
    result = results[0]

    # 3. Visualization magic
    # .plot() method draws all the boxes, class names, and confidence percentages
    # it returns a pixel matrix (numpy array) in BGR format, which is understood by OpenCV
    annotated_frame = result.plot(line_width = 5, font_size = 16)

    # 4. Encoding pixel matrix in format .jpg
    success, encoded_image = cv2.imencode(".jpg", annotated_frame)
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to encode image"
        )
    
    # 5. Return image in answer
    # Specify media_type="image/jpeg" so that Swagger/Browser understands that this is a photo, not text
    return Response(
        content=encoded_image.tobytes(),
        media_type="image/jpeg"
    )

@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """
    An endpoint for Celery background tasks status checking 
    """

    # Searching for a task in Redis by its ID
    task_result = AsyncResult(task_id, app=celery_app)

    # Forming basic answer
    response = {
        "task_id": task_id,
        "task_status": task_result.status,
    }

    # If task successfylly over, adding her results
    if task_result.status == "SUCCESS":
        response["result"] = task_result.result

    # If task fallen with error, deduce the reason
    elif task_result.status == "FAILURE":
        response["error"] = str(task_result.info)

    return response