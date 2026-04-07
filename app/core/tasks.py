import asyncio
from sqlalchemy import update
from app.database import async_session
from app.models import VideoRecord
import cv2
import logging
import subprocess
from pathlib import Path
from ultralytics import YOLO
from app.core.celery_app import celery_app

logger = logging.getLogger(__name__)

# Loading the weights globally once when the worker starts
model = YOLO("yolo11n.pt")

# Auto-retry parameter into decorator
@celery_app.task(
    bind=True,
    name="process_video_task",
    autoretry_for=(Exception,), # Catch all errors
    retry_kwargs={"max_retries": 3}, # 3 attempts
    retry_backoff=True # Pause between attempts will grow
)

@celery_app.task(bind=True, name="process_video_task")
def process_video_task(self, input_path: str, output_path: str):
    logger.info(f"Starting video processing: {input_path}")

    # Universal db update func
    async def update_db_status(new_status: str, path: str = None):
        try:
            async with async_session() as db:
                stmt = (
                    update(VideoRecord)
                    .where(VideoRecord.id == self.request.id)
                    .values(status=new_status, result_path=path)
                )
                await db.execute(stmt)
                await db.commit()
                logger.info(f"DB status for {self.request.id} updated to {new_status}")
        except Exception as db_err:
            logger.error(f"Failed to update DB for {self.request.id}: {db_err}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Failde to open video: {input_path}")
        raise FileNotFoundError(f"Cannot open video file: {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get video frames count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initializing the oven (VideoWriter) with the correct syntax
    temp_output = str(output_path).replace(".mp4", "_temp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    frames_counted = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=0.25)[0]
            annotated_frame = results.plot()

            out.write(annotated_frame)
            frames_counted += 1

            # Each 10 frames send progress to Redis
            if frames_counted % 10 == 0 or frames_counted == total_frames:
                # Zero devision defend
                safe_total = total_frames if total_frames > 0 else 1
                percent = int((frames_counted / safe_total) * 100)

                # Legal way to change task status
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'current': frames_counted,
                        'total': total_frames,
                        'percent': percent
                    }
                )

            # Logging process each 100 frames
            if frames_counted % 100 == 0:
                logger.info(f"Processed: {frames_counted} frames...")

        # LOOP HAS ENDED, THE FRAMES HAVE ENDED

        # IMPORTANT: First, close the OpenCV files so that they are saved to disk
        cap.release()
        out.release()

        # FFmpeg:
        logger.info("Starting FFmpeg conversion to H.264...")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", temp_output,
            "-vcodec", "libx264",
            "-f", "mp4", output_path
        ], check=True)

        # Delete temp file
        Path(temp_output).unlink(missing_ok=True)
        
        # Updating DB status -> completed
        asyncio.run(update_db_status(
            new_status="completed",
            path=str(output_path)
        ))

    except Exception as e:
        logger.error(f"Rendering fail: {e}. Worker will retry!")

        # Freeing resources on error if they are still open
        if 'cap' in locals() and cap.isOpened(): cap.release()
        if 'out' in locals(): out.release()

        # Checking worker retries
        if self.request.retries >= self.max_retries:
            logger.error(f"Task {self.request.id} failed definitively after {self.max_retries} retries")
            # Adding failed status to DB
            asyncio.run(
                update_db_status(new_status="failed")
            )
        # Send error forward so that Celery can retry
        raise e
    
    return {
        "status": "success",
        "output_path": str(output_path),
        "frames": frames_counted,
        "message": f"Video successfully processed. Total frames: {frames_counted}"
    }