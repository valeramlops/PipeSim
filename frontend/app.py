import streamlit as st
import requests
import os
import time
import pandas as pd
import numpy as np

# Page setting (Always first in Streamlit)
st.set_page_config(
    page_title="PipeSim MLOps",
    page_icon="🚀",
    layout="wide"
)

# Accessing the backend by service name in Docker (api)
BACKEND_URL = os.getenv("BACKEND_URL", "http://api:8000")

# Defender: API availability check
# Defend frontend from dumping, if backend turned off
def check_api_health():
    try:
        res = requests.get(f"{BACKEND_URL}/", timeout=2) 
        return res.status_code == 200
    except:
        return False
    
# Pinging server per each page loading
is_api_online = check_api_health()

# SideBar
with st.sidebar:
    st.title("System status")
    st.write("---")

    if is_api_online:
        st.success("API: Connected")
    else:
        st.error("API: Unavailable")
        st.warning("Check backend container (docker ps)")

    st.divider()
    if st.button("Reload interface", use_container_width=True):
        st.rerun()

# Main page header
st.title("PipeSim: Control panel")
st.write("Professional object detection system based on YOLOv11 and GPU acceleration")
st.divider()

# TABS
tab_photo, tab_video, tab_stats = st.tabs(["Photo", "Video", "Statistics and history"])

# TAB 1: PHOTO
with tab_photo:
    st.header("Image Detection")
    if not is_api_online:
        st.error("Function unavailable: there is no connection to the server")
    else:
        # Sub tabs
        sub_tab_instant, sub_tab_batch = st.tabs(["⚡ Instant (RAM-only)", "📦 Batch Mode (Save to DB)"])

        with sub_tab_instant:
            st.markdown("""
                <div style="
                    text-align: center;
                    padding: 1rem;
                    background-color: rgba(28, 131, 225, 0.1);
                    color: #8cb4df;
                    border-radius: 0.5rem;
                    border: 1px solid rgba(28, 131, 225, 0.2);
                    margin-bottom: 1rem;
                ">
                    📦 <b>Upload a single photo.</b> Inference will run instantly in RAM without saving to the database.
                </div>
                """,
                unsafe_allow_html=True
            )
            # Mode 1: Instant
            # Only photo loading widget
            uploaded_img = st.file_uploader("Upload single photo (JPG/PNG)", type=["jpg", "jpeg", "png"], key="single_upload")

            if uploaded_img:
                # Button
                analyze_clicked = st.button("Analyze photo", type="primary", use_container_width=True, key="btn_single")

                # Divide the screen into two columns for beauty
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Original")
                    st.image(uploaded_img, use_container_width=True)

                with col2:
                    st.subheader("Result")

                    # Logic inside the right column:
                    if not analyze_clicked:
                        # Until you click, we're showing the plate
                        st.markdown(
                            """
                            <div style=
                            "text-align: center;
                            padding: 1rem;
                            background-color: rgba(28, 131, 225, 0.1);
                            color: #8cb4df;
                            border-radius: 0.5rem;
                            border: 1px solid rgba(28, 131, 225, 0.2);
                            margin-bottom: 1rem;">
                                👆 Click the button above to start detection
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        with st.spinner("Processing in RAM..."):
                            try:
                                # Packing file for FastAPI
                                files = {
                                    "file": (uploaded_img.name, uploaded_img.getvalue(), uploaded_img.type)
                                }
                                
                                res = requests.post(f"{BACKEND_URL}/api/vision/predict_image", files=files)

                                if res.status_code == 200:
                                    # Server return bytes of ready image, Streamlit drawing it directly
                                    st.image(res.content, use_container_width=True)
                                    st.toast("Photo successfully processed", icon='⚡')
                                else:
                                    st.error(f"Server rejected request. Code: {res.status_code}")
                            
                            except Exception as e:
                                st.error(f"Network error: {e}")

        with sub_tab_batch:
            # markdow
            st.markdown(
                """
                <div style="
                    text-align: center;
                    padding: 1rem;
                    background-color: rgba(28, 131, 225, 0.1);
                    color: #8cb4df;
                    border-radius: 0.5rem;
                    border: 1px solid rgba(28, 131, 225, 0.2);
                    margin-bottom: 1rem;
                ">
                    📦 <b>Upload multiple photos.</b> Inference will run in batch, and JSON results will be saved to PostgreSQL. 
                </div>
                """,
                unsafe_allow_html=True
            )

            # Accept_multiple_files=True Allows you to select many files at once
            uploaded_imgs = st.file_uploader(
                "Upload photos batch (JPG/PNG)",
                type = ["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="batch_upload_field"
            )

            if uploaded_imgs:
                st.write(f"Selected files: {len(uploaded_imgs)}")

                if st.button("Process Batch 📦", type="primary", use_container_width=True, key="btn_multi"):
                    with st.spinner(f"Sending {len(uploaded_imgs)} files to backend..."):
                        try:
                            # Create a list of tuples with the "files" key,
                            # as FastAPI expects the parameter files:List[UploadFile]
                            files_payload = [
                                ("files", (f.name, f.getvalue(), f.type)) for f in uploaded_imgs
                            ]

                            res = requests.post(f"{BACKEND_URL}/api/vision/upload", files=files_payload)

                            if res.status_code == 200:
                                data = res.json()

                                # Adding data in session memory
                                st.session_state['batch_results'] = data.get("results", [])

                                st.success("Batch processed!")
                                st.toast(f"✅ {data.get('processed_count')} photos pushed to DB!", icon='📦')

                            else:
                                st.error(f"Server rejected request. Code: {res.status_code}")
                                st.json(res.json())

                        except Exception as e:
                            st.error(f"Network error: {e}")
                
                # Reading data from session
                if 'batch_results' in st.session_state and st.session_state['batch_results']:
                    st.divider()
                    st.subheader("Batch Results")

                    for item in st.session_state['batch_results']:
                        with st.expander(f"📄 {item['original_filename']} (Objects: {len(item['detections'])})"):

                            # Showing image
                            if item.get("processed_url"):
                                full_img_url = f"{BACKEND_URL}{item['processed_url']}"
                                # Downloads bytes like in history
                                try:
                                    img_response = requests.get(full_img_url)
                                    if img_response.status_code == 200:
                                        st.image(img_response.content, use_container_width=True)
                                    else:
                                        st.error(f"Backend did not return the image. Error: {img_response.status_code}")
                                        st.code(f"Tried to download through link: {full_img_url}")
                                except Exception as e:
                                    st.error(f"Failed to load image from backend. Code: {e}")
                            else:
                                st.warning("Visual results not found")

                            # Checkbox for JSON
                            if st.checkbox("Show raw JSON", key=f"chk_json_{item['detection_id']}"):
                                st.json(item['detections'])

# TAB 2: Video page
with tab_video:
    st.header("Stream analysis")

    # Blocking downloads when the API crashes
    if not is_api_online:
        st.error("Server temporarily unavailable. File loading is turned off")
    else:
        # File upload widget with format restrictions
        uploaded_file = st.file_uploader("Choose video", type=["mp4", "avi", "mov"], key="video_upload")
    
        if uploaded_file:
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
                            task_id = data.get("task_id")
                            st.success("Video in queue. Monitoring...")

                            st.write("### Processing status")

                            # Empty containers for interface updating (Polling)
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            # Status check every 2 seconds loop
                            while True:
                                task_res = requests.get(f"{BACKEND_URL}/api/vision/task/{task_id}").json()
                                status = task_res.get("task_status")
                                progress = task_res.get("progress", 0)

                                if status == "PENDING":
                                    status_text.warning("Task in queue. Waiting for worker...")

                                elif status == "PROGRESS":
                                    progress_bar.progress(int(progress))
                                    status_text.info(f"Frame detection: {int(progress)}%")

                                elif status == "SUCCESS":
                                    progress_bar.progress(100)
                                    status_text.success("Processing is over")
                                    st.toast("Video is ready to watch!")

                                    # Parsing filename and collecting url for download
                                    raw_result = task_res.get("result")
                                    # If worker return dict, getting path by key
                                    if isinstance(raw_result, dict):
                                        result_path = raw_result.get("output_path", "")
                                    else:
                                        result_path = str(raw_result)
                                    filename = os.path.basename(result_path)
                                    video_url = f"{BACKEND_URL}/static/videos/{filename}"

                                    st.write("### Results")

                                    # Loading video bytes in Docker-net
                                    video_bytes = requests.get(video_url).content
                                    st.video(video_bytes) # Rendering player

                                    break # Loop end
                            
                                elif status == "FAILURE":
                                    status_text.error(f"Worker error: {task_res.get('error')}")
                                    break
                                    
                                # Pause
                                time.sleep(2)

                        else:
                            # If backend return 400 or 422 (validation error)
                            st.error(f"Server rejected request. Code: {response.status_code}")
                            st.json(response.json()) # Showing gray error for debugging
                    
                    except Exception as e:
                        # If network is down (timeout or Docker dump)
                        st.error(f"Internal network error: {e}")

# TAB 3: Statistic and History
with tab_stats:
    st.header("System analysis")

    if not is_api_online:
        st.warning("Failed to load history: API doesn't respond")
    else:
        # 1. Dashboard (Statistic)
        try:
            hist_res = requests.get(f"{BACKEND_URL}/api/vision/history", timeout=5)
            if hist_res.status_code == 200:
                data = hist_res.json()
                history_data = data.get("history", [])
                count = data.get("count", 0)

                col1, col2, col3 = st.columns(3)
                col1.metric("Total processed", count, "Database (PostgreSQL)")
                col2.metric("Acceleration of inference", "CUDA", "GPU Active")
                col3.metric("Worker status", "Online", "Celery", delta_color="normal")
                st.divider()

                # A stub for future metrics from Block 7
                st.caption("GPU load (Simulation before Prometheus/Grafana implementation)")
                chart_data = pd.DataFrame(np.random.randn(20, 1) * 10 + 50, columns=["Disposal GPU (%)"])
                st.line_chart(chart_data)
                st.divider()

                # 2. History log
                st.subheader("Operation history")

                if not history_data:
                    st.info("Database is empty now")
                else:
                    # Trying process data from DB (if request to API successfully done)
                    for record in history_data:
                        # Lowercase the status to avoid comparison errors (Success vs success)
                        db_status = record.get('status', '').lower()

                        # Icon choise logic: completed, processing or dump with error
                        if db_status in ["success", "completed"]:
                            status_icon = "✅"
                        elif db_status in ["processing", "pending", "progress"]:
                            status_icon = "⏳"
                        else:
                            status_icon = "❌"
                        # Creating expander. In header - filename and status
                        with st.expander(f"{status_icon} {record['filename']} ({record['status'].upper()})"):
                            st.write(f"Task ID: {record['task_id']}")
                            st.write(f"Data: {record['created_at']}")

                            # If video successfully processed and result path is exist in DB - trying to show player
                            if db_status in ["success", "completed"] and record.get('result_url'):
                                # Forming full link on static file through backend proxy-URL
                                full_url = f"{BACKEND_URL}{record['result_url']}"
                                st.write("---")
                                try:
                                    # Download bytes through link
                                    file_bytes = requests.get(full_url).content

                                    # Check the extensions: if it is a picture, draw it immediately
                                    if record['filename'].lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                                        st.image(file_bytes, use_container_width=True)

                                    # if video - hide player button
                                    else:
                                        if st.button(f"Load Video", key=f"btn_{record['task_id']}"):
                                            st.video(file_bytes)
                                except Exception as e:
                                    st.error("Failed to upload file from server")
            else:
                # If backend return 404, 500 or other error
                st.error(f"History request error: {hist_res.status_code}")
                        
        except Exception as e:
            # If API container is turned off or the Docker network times out
            st.error(f"Network error or failed to process database data: {e}")