import asyncio
from sqlalchemy import update
from app.database import async_session
from app.models import VideoRecord
import cv2
import subprocess
import gc
from pathlib import Path
from ultralytics import YOLO
from app.core.celery_app import celery_app
from app.core.logger import logger
from app.core.llm_service import generate_video_report
from app.database import SessionLocal

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

def process_video_task(self, input_path: str, output_path: str):
    logger.info(f"Starting video processing: {input_path}")

    # Universal db update func
    def update_db_status(new_status: str, path: str = None, detections: dict = None):
        async def _async_update():
            async with async_session() as db:
                await db.execute(
                    update(VideoRecord)
                    .where(VideoRecord.id == self.request.id)
                    .values(status=new_status, result_path=path, detections_data = detections)
                )
                await db.commit()

        try:
            asyncio.run(_async_update())
            logger.info(f"DB status for {self.request.id} updated to {new_status}")
        except Exception as db_err:
            logger.error(f"Failed to update DB for {self.request.id}: {db_err}")

    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {input_path}")
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

        # Statistic initialization
        stats = {
            "total_frames": 0,
            "class_counts": {},
            "timeline": []
        }
        frames_counted = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=0.25, device=0)[0]
            annotated_frame = frame.copy()

            current_frame_objects = 0

            if results.boxes is not None:
                for box in results.boxes:
                    # Getting coords, class and conf
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]

                    # Data collecting
                    stats["class_counts"][class_name] = stats["class_counts"].get(class_name, 0) + 1
                    current_frame_objects += 1

                    label = f"{class_name} {conf:.2f}"

                    # Setting style (In OpenCV BGR format NOT RGB)
                    color = (255, 144, 30)
                    text_color = (255, 255, 255)

                    # 1. Drawing slim boarder
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                    # 2. Doing filled label for text
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated_frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)

                    # 3. Writting text with anti-aliasing (font smoothing LINE_AA)
                    cv2.putText(
                        annotated_frame, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA
                    )

            out.write(annotated_frame)

            # Writing dot on chart (every 10th frame)
            if frames_counted % 10 == 0:
                stats["timeline"].append({
                    "frame": frames_counted,
                    "count": current_frame_objects
                })

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

        # Writing total frames count in stats
        stats["total_frames"] = frames_counted

        gc.collect()

        # FFmpeg:
        logger.info("Starting FFmpeg conversion to H.264...")
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", temp_output,
                "-vcodec", "libx264",
                # Suppressing FFmpeg's extra noise to avoid clogging the loguru logs
                "-loglevel", "error",
                "-f", "mp4", output_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            logger.success("FFmpeg conversion successful")
        except subprocess.CalledProcessError as ffmpeg_err:
            logger.error(f"FFmpeg failed with error: {ffmpeg_err.stderr.decode()}")
            raise RuntimeError("FFmpeg conversion failed")

        # Delete temp file
        Path(temp_output).unlink(missing_ok=True)

        # Generate report through LLM
        logger.info("Video processed. Sending statistics to Gemini API...")

        # Calling service
        llm_text_report = generate_video_report(stats)

        # Adding text directly to stats dict
        stats["llm_summary"] = llm_text_report

        logger.info("LLM report successfully added to results")

        # Updating DB status -> completed
        update_db_status(
            new_status="completed",
            path=str(output_path),
            detections=stats
        )

        return {
            "status": "success",
            "output_path": str(output_path),
            "frames": frames_counted,
            "llm_summary": stats.get("llm_summary", ""),
            "message": f"Video successfully processed. Total frames: {frames_counted}"
        }

    except Exception as e:
        logger.error(f"Rendering fail: {e}. Worker will retry!")

        # Freeing resources on error if they are still open
        if 'cap' in locals() and cap.isOpened(): cap.release()
        if 'out' in locals(): out.release()

        # Checking worker retries
        if self.request.retries >= self.max_retries:
            logger.error(f"Task {self.request.id} failed definitively after {self.max_retries} retries")
            # Adding failed status to DB
            update_db_status(new_status="failed")

        # Send error forward so that Celery can retry
        raise e