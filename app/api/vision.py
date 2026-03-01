from fastapi import APIRouter, UploadFile, File

router = APIRouter()

@router.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    """
    An endpoint for loading images
    Here would work ML-model in future
    """
    # Return info about loaded file
    return {
        "filename": image.filename,
        "content_type": image.content_type,
        "message": "File received successfully! Ready for Computer Vision"
    }