import pytest
import uuid
import cv2
import numpy as np
from httpx import AsyncClient
from unittest.mock import patch

from app.main import app
from app.database import get_db
from app.models import VideoRecord

# BLOCK 1: File validation (400 Errors)

# Test 1: Checking if an image is uploaded in the wrong format
async def test_upload_image_invalid_extension(async_client: AsyncClient):
    # Create fake txt file in memory
    files = {
        "files": ("hack.txt", b"print('hacked')", "text/plain")
    }

    # Send POST request to batch-upload endpoint
    response = await async_client.post("/api/vision/upload", files=files)

    # Expect the server to tell us to go away (code 400)
    assert response.status_code == 400

    # Checking that the server returns the correct error message
    data = response.json()
    assert "detail" in data
    assert data["detail"] == "No valid images uploaded in the batch"

# Test 2: Checking wrong video format uploading
async def test_upload_video_invalid_extension(async_client: AsyncClient):
    files = {
        "file": ("virus.exe", b"malicious code", "application/x-msdownload")
    }

    # Send POST request
    response = await async_client.post("/api/vision/upload_video", files=files)

    # Expect code 400 Bad Request
    assert response.status_code == 400
    assert "Supporting only .mp4, .avi, .mov" in response.json()["detail"]

# BLOCK 2: Database and routing

# Test 3: Checking DELETE method
async def test_delete_history_record_format(async_client: AsyncClient):
    # Sending fake UUID
    fake_uuid = "123e4567-e89b-12d3-a456-426614174000"

    # If the routing works correctly (task_id is written correctly)
    # should get a 404 Not Found (there is no record in the mock database) or a 500 (the mock database is not configured)
    # but most importantly, not a 422 Unprocessable Entity
    response = await async_client.delete(f"/api/vision/history/{fake_uuid}")

    assert response.status_code != 422

# Test 4: Checking whether the history is retrieved from the database (SQLite in-memory)
async def test_get_video_history(async_client: AsyncClient):
    
    # Getting mock database session
    db_generator = app.dependency_overrides[get_db]()
    db = await anext(db_generator)

    # Put fake record
    fake_id = str(uuid.uuid4())
    fake_record = VideoRecord(
        id = fake_id,
        filename="test_video.mp4",
        status="completed",
        result_path="/fake/path/res.mp4"
    )
    db.add(fake_record)
    await db.commit()

    # Pulling API
    response = await async_client.get("/api/vision/history")

    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 1
    assert data["history"][0]["filename"] == "test_video.mp4"

# BLOCK 3: ML and Celery (Mocking)

# Test 5: Checking successful image processing (Photo generation in RAM)
async def test_predict_image_success(async_client: AsyncClient):
    # Generating black square 100x100
    black_image = np.zeros((100, 100, 3), dtype=np.uint8)
    success, encoded_img = cv2.imencode('.jpg', black_image)
    assert success, "Failed to encode dummy image"

    files = {
        "file": ("test_black.jpg", encoded_img.tobytes(), "image/jpeg")
    }

    response = await async_client.post("/api/vision/predict_image", files=files)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 0

# Test 6: Checking the upload of a video with a Celery stub
@patch("app.main.vision.process_video_task.delay")
async def test_upload_video_success(mock_delay, async_client: AsyncClient):
    # Setting stub
    class FakeTask:
        id = "fake-task-123"
    mock_delay.return_value = FakeTask()

    files = {
        "file": ("test_vid.mp4", b"fake video bytes", "video/mp4")
    }

    response = await async_client.post("/api/vision/upload_video", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == "fake-task-123"

    # Check that Celery was actually called
    mock_delay.assert_called_once()

# BLOCK 4 Edge Cases

# Test 7: History request, when database is empty
async def test_get_video_history_empty(async_client: AsyncClient):
    # Dont add fake entries, the database is virgin
    response = await async_client.get("/api/vision/history")

    assert response.status_code == 200
    data = response.json()

    # Check that API is reacts adequately to emptiness
    assert data["status"] == "success"
    assert data["count"] == 0
    assert data["history"] == [] # An empty list should be returned

# Test 8: Simulating a database crash (Checking for a 500 error)
async def test_get_video_history_db_error(async_client: AsyncClient):
    # 1. Preserving the original working mock-dependency from conftest
    original_override = app.dependency_overrides.get(get_db)

    # 2. Creating "evil" dependency, which simulating disconnection
    async def evil_get_db():
        class EvilSession:
            async def execute(self, *args, **kwargs):
                raise RuntimeError("Database server is on fire")
        yield EvilSession()
    
    # 3. Replace the normal database with an "evil" database
    app.dependency_overrides[get_db] = evil_get_db

    # 4. Make request (expect the router to catch the RuntimeError and return 500)
    response = await async_client.get("/api/vision/history")

    assert response.status_code == 500
    assert response.json()["detail"] == "Database error"

    # 5. IMPORTANT: Returning the normal dependency back to not break other tests
    app.dependency_overrides[get_db] = original_override