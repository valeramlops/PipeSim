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

st.title("PipeSim: control panel")
st.write("Welcome to MLOps simulator interface. File uploading and working with Computer Vision will be available soon")

st.divider()

st.subheader("System status")

try:
    # Test request to backend
    response = requests.get(f"{BACKEND_URL}/", timeout=5)

    if response.status_code == 200:
        st.success(f"Successfully connect to backend. URL: {BACKEND_URL}")

        # Checking db (Video history)
        try:
            history_resp = requests.get(f"{BACKEND_URL}/api/vision/history", timeout=2)
            if history_resp.status_code == 200:
                count = history_resp.json().get("count", 0)
                st.info(f"DataBase in touch. Video in history: {count}")
        except:
            pass # If history not answered - not critical for this test

    else:
        st.warning(f"Backend answered, but return code: {response.status_code}")
except requests.exceptions.ConnectionError:
    st.error(f"Connecting error! Frontend not see backend on url: {BACKEND_URL}. Check networks in docker-compose")
    