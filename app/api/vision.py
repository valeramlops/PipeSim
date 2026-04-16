from fastapi import APIRouter, UploadFile, File, HTTPException, Response, BackgroundTasks, Request
import shutil
import uuid
from pathlib import Path
from ultralytics import YOLO
import cv2
from typing import List
import logging
import numpy as np

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db, async_session
from app.models import DetectionRecord, VideoRecord
from app.core.tasks import process_video_task
from app.core.celery_app import celery_app
from celery.result import AsyncResult

router = APIRouter()

# Create folder path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOAD_VIDEO_DIR = BASE_DIR / "uploads" / "videos"

print(f"DEBUG: Static files directory is {UPLOAD_VIDEO_DIR}")

# Automation create folder if not exists
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
        file_path = UPLOAD_VIDEO_DIR / unique_filename

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

    # Write into db
    try:
        async with async_session() as db:
            new_video = VideoRecord(
                id = task.id, #Binding record in DB to task ID in Celery
                filename=file.filename,
                status = "processing"
            )
            db.add(new_video)
            await db.commit()
    except Exception as e:
        logger.error(f"Failed to save video record to DB: {e}")

    # 3. Instantly responding to the user
    return {
        "status": "processing",
        "video_id": video_id,
        "task_id": task.id,
        "message": "Video successfully saved. Processing started in the background"
    }

@router.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    """
    Synchronus image processing strictly in RAM (No disk saving)
    Returns image with drawn bounding boxes
    """

    valid_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    if not file.filename.lower().endswith(valid_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supporting only images."
        )

    # 1. Reading bytes uploaded file directly in RAM
    contents = await file.read()

    # 2. Convert bytes into OpenCV format (BGR pixel matrix)
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(
            status_code=400,
            detail="Failed to decode image"
        )
    
    # 3. Instant pixel matrix prediction
    results = model(img, conf=0.25)
    result = results[0]

    # 4. Visualization: .plot() method drawing borders and classes
    annotated_frame = result.plot(line_width=5, font_size=16)

    # 5. Encode back into .jpg to sending through net
    success, encoded_image = cv2.imencode(".jpg", annotated_frame)
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to encpode image"
        )

    # 6. Return bytes
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
        "progress": 0
    }

    # If task in progress, get percents from meta dict
    if task_result.status == "PROGRESS":
        response["progress"] = task_result.info.get("percent", 0)

    # If task successfylly over, adding her results
    if task_result.status == "SUCCESS":
        response["progress"] = 100
        response["result"] = task_result.result

    # If task fallen with error, deduce the reason
    elif task_result.status == "FAILURE":
        response["error"] = str(task_result.info)

    return response

@router.get("/history")
async def get_video_history():
    """
    Return video history from PostgreSQL
    """
    try:
        async with async_session() as db:
            stmt = select(VideoRecord).order_by(VideoRecord.created_at.desc())
            result = await db.execute(stmt)
            records = result.scalars().all()

            return {
                "status": "success",
                "count": len(records),
                "history": [
                    {
                        "task_id": r.id,
                        "filename": r.filename,
                        "status": r.status,
                        "result_url": f"/static/videos/{Path(r.result_path).name}" if r.result_path else None,
                        "created_at": r.created_at
                    } for r in records
                ]
            }
    except Exception as e:
        logger.error(f"History error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Database error"
        )