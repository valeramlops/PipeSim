from fastapi import APIRouter, UploadFile, File, HTTPException, Response
import shutil
import uuid
from pathlib import Path
from ultralytics import YOLO
import cv2

router = APIRouter()

# Create folder path
UPLOAD_DIR = Path("uploads/images")

# Automation create folder if not exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Model init (nano for quick start on CPU)
# Model is loaded once on server start
model = YOLO("yolo11n.pt")

@router.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    """
    Endpoint for uploading images and real-time detection using YOLOv11
    """
    # 1. Security: check file format
    valid_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    is_image_mime = image.content_type.startswith("image/")
    is_valid_ext = image.filename.lower().endswith(valid_extensions)
    if not (is_image_mime or is_valid_ext):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. MIME: {image.content_type}"
        )
    
    # 2. Generating unique filename
    unique_filename = f"{uuid.uuid4()}_{image.filename}"
    file_path = UPLOAD_DIR / unique_filename

    # 3. Safe file on disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # 4. Run through the AI
    # conf=0.25 discards garbage predictions (we only keep what the AI confidence is 25%+)
    results = model(str(file_path), conf=0.25)

    # Getting results
    real_detections = []

    # YOLO can process batches (pack) of images, so it returns a list.
    # We passed a single image, so we take the result from the results
    result = results[0]
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

    # 5. Return API
    return {
        "status": "success",
        "original_filename": image.filename,
        "path": str(file_path),
        "detections": real_detections,
        "message": "Image processed with YOLOv11"
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