import pytest
from unittest.mock import patch, MagicMock
from app.core.tasks import process_video_task

# Block 1: Error test (Negative test)

# Test 1: What happens if you upload a non-existen video?
@patch("app.core.tasks.cv2.VideoCapture")
def test_process_video_file_not_found(mock_cv2_cap):
    # Configure the stub so that it says "I couldn't open the file"
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_cv2_cap.return_value = mock_cap

    # Since tha task has bind = True, it expects the 'self' object as the first argument
    process_video_task.request.id = "fake-task-id-123"
    process_video_task.request.retries = 0
    process_video_task.max_retries = 3

    # Checking that the function actually throws a FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Cannot open video file"):
        process_video_task("bad_path.mp4", "out.mp4")
    
# Block 2: The Perfect Path (Total mocking)

# Test 2: Successfully video processing
@patch("app.core.tasks.model") # Silencing the YOLO neural network
@patch("app.core.tasks.cv2.VideoCapture") # Silencing video read
@patch("app.core.tasks.cv2.VideoWriter") # Silencing video write
@patch("app.core.tasks.subprocess.run") # Silencing FFmpeg
@patch("app.core.tasks.asyncio.run") # Silencing record in database
def test_process_video_success(mock_asyncio, mock_subprocess, mock_writer, mock_capture, mock_yolo):
    # 1. Fake worker (self)
    process_video_task.update_state = MagicMock()
    process_video_task.request.id = "fake-task-id-123"

    # 2. Fake video
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 1 # Let function think that there is 1 frame, 1 fps, etc

    # Configure .read() method so that it outputs 1 frame and then stops
    fake_frame = MagicMock()
    mock_cap.read.side_effect = [
        (True, fake_frame), # First pass of the loop: reading the frame
        (False, None) # Second pass of the loop: frames are over, exit the loop
    ]
    mock_capture.return_value = mock_cap

    # 3. Fake YOLO AI
    mock_result = MagicMock()
    mock_result.boxes = None # Pretend that nothing is found on the frame (to avoid complicating the test)
    mock_yolo.return_value = [mock_result]

    # Starting task
    result = process_video_task("input.mp4", "output.mp4")

    # Checking that function has completed and returned the correct dictionary
    assert result["status"] == "success"
    assert result["frames"] == 1
    assert result["output_path"] == "output.mp4"

    # Checking that FFmpeg was called 1 time
    mock_subprocess.assert_called_once()

    # Checking that the status 'completed' was attempted to be written to the database
    mock_asyncio.assert_called()

# BLOCK 3: Fatal error

# Test 3: Exceeding the attempt limit
@patch("app.core.tasks.cv2.VideoCapture")
@patch("app.core.tasks.asyncio.run") # Silencing the database to intercept the call
def test_process_video_fatal_failure(mock_asyncio, mock_cv2_cap):
    # Stub: the file does not open
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_cv2_cap.return_value = mock_cap

    # Set up the task as if it's your last 3rd attempt
    process_video_task.request.id = "fatal-task-123"
    process_video_task.request.retries = 3
    process_video_task.max_retries = 3

    # Function anyway must return error
    with pytest.raises(FileNotFoundError):
        process_video_task("corrupted.mp4", "out.mp4")

    # BUT at the same time, she had to call asyncio.run to record the failed status
    mock_asyncio.assert_called()