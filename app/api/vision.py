from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import uuid
from pathlib import Path

router = APIRouter()

# Create folder path
UPLOAD_DIR = Path("uploads/images")

# Automation create folder if not exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    """
    An endpoint for loading and safe images
    """
    # 1. Security: check file format
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    # 2. Generating unique filename
    unique_filename = f"{uuid.uuid4()}_{image.filename}"
    file_path = UPLOAD_DIR / unique_filename

    # 3. Safe file on disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # 4. Return file path
    return {
        "original_filename": image.filename,
        "saved_filename": unique_filename,
        "content_type": image.content_type,
        "path": str(file_path),
        "message": "File successfully saved to disk!"
    }