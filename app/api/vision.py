from fastapi import APIRouter, UploadFile, File, HTTPException, Response, BackgroundTasks, Request, Depends
import shutil
import uuid
from pathlib import Path
from ultralytics import YOLO
import cv2
from typing import List
import numpy as np
import aiofiles

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db, async_session
from app.models import VideoRecord
from app.core.tasks import process_video_task
from app.core.logger import logger
from app.core.celery_app import celery_app
from celery.result import AsyncResult

router = APIRouter()

# Create folders paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOAD_VIDEO_DIR = BASE_DIR / "uploads" / "videos"
UPLOAD_IMAGE_DIR = BASE_DIR / "uploads" / "images"

# Automation create folders if not exists
UPLOAD_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO("yolo11n.pt")

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
                VideoRecord(
                    id=rec["id"],
                    filename=rec["filename"],
                    status="completed",
                    result_path=rec["result_path"]
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
        file_path = UPLOAD_IMAGE_DIR / unique_filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        saved_files_info.append({
            "id": detection_id,
            "filename": file.filename,
            "path": str(file_path),
            "unique_filename": unique_filename
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

    # Getting results and drawing
    for i, result in enumerate(results):
        file_info = saved_files_info[i]

        # Drawing
        annotated_frame = result.plot(line_width = 3)

        # Generating name for processed file
        res_filename = f"res_{file_info['unique_filename']}"
        res_path = UPLOAD_IMAGE_DIR / res_filename
        
        # Save on disk
        cv2.imwrite(str(res_path), annotated_frame)

        real_detections = []
        for box in result.boxes:
            coords = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            real_detections.append({
                "class": class_name,
                "confidence": round(conf, 2),
                "bbox": [round(c, 1) for c in coords]
            })

        # Create answer with link on image
        processed_url = f"/static/images/{res_filename}"

        # Prepare data for json-answer to user
        response_data.append({
            "detection_id": file_info["id"],
            "original_filename": file_info["filename"],
            "processed_url": processed_url,
            "detections": real_detections
        })

        # Filling the array for background recording in the database
        db_records_to_save.append({
            "id": file_info["id"],
            "filename": file_info["filename"],
            "detections": real_detections,
            "result_path": str(res_path)
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

    # Chunk size (5 mb)
    CHUNK_SIZE = 1024 * 1024 * 5

    async with aiofiles.open(video_path, "wb") as buffer:
        while chunk := await file.read(CHUNK_SIZE):
            await buffer.write(chunk)

    logger.info(f"Video {unique_filename} successfully saved via streaming")

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
    Synchronus image processing strictly in RAM (with disk mirroring)
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

    # 1.1 Saving original on disk
    # Genereting one ID for original and result
    prediction_id = str(uuid.uuid4())

    orig_filename = f"ram_orig_{prediction_id}_{file.filename}"
    orig_path = UPLOAD_IMAGE_DIR / orig_filename

    # Instant through bytes on disk
    async with aiofiles.open(orig_path, "wb") as buffer:
        await buffer.write(contents)
    logger.success(f"Original image saved to disk: {orig_filename}")

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

    # 4.1 Save on disk (processed photo NOT ORIGINAL)

    res_filename = f"ram_res_{prediction_id}_{file.filename}"
    res_path = UPLOAD_IMAGE_DIR / res_filename

    success_save = cv2.imwrite(str(res_path), annotated_frame)
    if success_save:
        logger.success(f"Processed image saved to disk: {res_filename}")
    else:
        logger.error(f"CRITICAL: Failed to write image to {res_path}")

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
async def get_video_history(db: AsyncSession = Depends(get_db)):
    """
    Return video history from PostgreSQL
    """
    try:
        # Looking for record in database
        stmt = select(VideoRecord).order_by(VideoRecord.created_at.desc())
        result = await db.execute(stmt)
        records = result.scalars().all()

        # Return info
        return {
            "status": "success",
            "count": len(records),
            "history": [
                {
                    "task_id": r.id,
                    "filename": r.filename,
                    "status": r.status,
                    "result_url": (
                        f"/static/{'images' if r.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')) else 'videos'}/{Path(r.result_path).name}"
                        if r.result_path else None
                    ),
                    "created_at": r.created_at
                } for r in records
            ]
        }
        # Return error
    except Exception as e:
        logger.error(f"History error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Database error"
        )
    
@router.delete("/history/{task_id}")
async def delete_history_record(task_id: str):
    """
    Delete record from Database and physically erases the file from the disk
    """
    try:
        async with async_session() as db:
            # 1. Looking for record in database
            stmt = select(VideoRecord).where(VideoRecord.id == task_id)
            result = await db.execute(stmt)
            record = result.scalar_one_or_none()

            if not record:
                raise HTTPException(
                    status_code=404,
                    detail="Record not found in database"
                )
            
            # 2. Deleting physical file (if it exists)
            if record.result_path:
                file_path = Path(record.result_path)
                if file_path.exists():
                    file_path.unlink() # Safety deleting file
                    logger.info(f"Deleted physical file: {file_path.name}")

            # 3. Deleting record from PostgreSQL
            await db.delete(record)
            await db.commit()

            logger.success(f"Task {task_id} completely removed from system")
            return {
                "status": "success",
                "message": "Record and file completely deleted"
            }
    
    except Exception as e:
        logger.error(f"Failed to delete record {task_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Database or FileSystem error"
        )