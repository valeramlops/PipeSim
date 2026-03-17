import cv2
import logging
from ultralytics import YOLO
from app.core.celery_app import celery_app

logger = logging.getLogger(__name__)

# Loading the weights globally once when the worker starts
model = YOLO("yolo11n.pt")

@celery_app.task(bind=True, name="process_video_task")
def process_video_task(self, input_path: str, output_path: str):
    logger.info(f"Starting video processing: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Failde to open video: {input_path}")
        return {
            "status": "error",
            "error": "Cannot open video file"
        }
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get video frames count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initializing the oven (VideoWriter) with the correct syntax
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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
            
    except Exception as e:
        logger.error(f"Rendering fail: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
    
    finally:
        cap.release()
        out.release()
        logger.info(f"Video saved: {output_path}. Frames: {frames_counted}")
    
    return {
        "status": "success",
        "output_path": str(output_path),
        "frames": frames_counted,
        "message": f"Video successfully processed. Total frames: {frames_counted}"
    }