import streamlit as st
import requests
import os

# Accessing the backend by service name in Docker (api)
BACKEND_URL = os.getenv("BACKEND_URL", "http://api:8000")

# Page settings
st.set_page_config(
    page_title="PipeSim MLOps",
    page_icon="🚀",
    layout="wide"
)

# SideBar
with st.sidebar:
    st.title("System Status")
    st.write("---")
    try:
        # Fust backend ping for indicator
        res = requests.get(f"{BACKEND_URL}/", timeout=2)
        if res.status_code == 200:
            st.success("API: Connected")
        else:
            st.warning(f"API: Error {res.status_code}")
    except:
        st.error("API: There is no connection")

# Main page
st.title("PipeSim: Computer Vision")
st.write("Load video for object detection. Video will be sent to the Celery queue for asynchronous YOLOv11 processing")
st.divider()

# File upload widget with format restrictions
uploaded_file = st.file_uploader("Choose videofile", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Calculating file size for beauty of the display
    file_size_mb = round(uploaded_file.size / (1024 * 1024), 2)
    st.info(f"File ready: **{uploaded_file.name}** ({file_size_mb} MB)")

    # Send to backend button
    if st.button("Send to processing", type="primary", use_container_width=True):

        # Spinner spins while we wait for a response from FastAPI
        with st.spinner("Sending file to backend..."):
            try:
                # 1. Pack the file in a format that FastAPI understands (UploadFile)
                # Tuple structure: (fle_name, binary_data, MIME_type)
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }

                # 2. Send POST request to ready endpoint
                upload_endpoint = f"{BACKEND_URL}/api/vision/upload_video"
                response = requests.post(upload_endpoint, files=files)

                # 3. Answer processing
                if response.status_code == 200:
                    data = response.json()
                    st.success("Video successfully uploaded on server and sended to worker")

                    # Return task ID
                    col1, col2 = st.columns(2)
                    col1.metric("Task ID (Celery)", data.get("task_id", "N/A"))
                    col2.metric("Video ID (DB)", data.get("video_id", "N/A"))

                else:
                    # If backend return 400 or 422 (validation error)
                    st.error(f"Server rejected file. Code: {response.status_code}")
                    st.json(response.json()) # Showing gray error for debagging
            
            except Exception as e:
                # If network is down (timeout or Docker dump)
                st.error(f"Internal network error: {e}")