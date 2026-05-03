from locust import HttpUser, task, between
from pathlib import Path
from loguru import logger

class PipeSimStressUser(HttpUser):
    # Virtual user make pause from 1 to 3 second between actions
    wait_time = between(1, 3)

    @task(1)
    def check_health(self):
        # Common server ping
        self.client.get("/")
    
    @task(3) # '3' means that this task will be called 3 times more often
    def analyze_instant_image(self):
        # Simulate sending a frame from the camera to the instant inference
        # Take any picture from test data
        image_path = Path("test_image.jpg")

        fake_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, line Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json"
        }

        # Object-oriented file existence check
        if image_path.exists():
            with image_path.open("rb") as img:
                self.client.post(
                    "/api/vision/predict_image",
                    files={
                        "file": (image_path.name, img, "image/jpeg")
                    },
                    headers=fake_headers
                )
        else:
            logger.info(f"Error: File not found on path: {image_path.absolute()}")