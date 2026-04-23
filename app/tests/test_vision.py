import pytest
from httpx import AsyncClient

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

# Test 3: Checking DELETE method
async def test_delete_history_record_format(async_client: AsyncClient):
    # Sending fake UUID
    fake_uuid = "123e4567-e89b-12d3-a456-426614174000"

    # If the routing works correctly (task_id is written correctly)
    # should get a 404 Not Found (there is no record in the mock database) or a 500 (the mock database is not configured)
    # but most importantly, not a 422 Unprocessable Entity
    response = await async_client.delete(f"/api/vision/history/{fake_uuid}")

    assert response.status_code != 422