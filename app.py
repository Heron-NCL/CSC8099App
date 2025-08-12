# app.py — Streamlit Community Cloud 
import os
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
os.environ.setdefault("OPENCV_FOR_THREAD_NUMBER", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av
import torch
import pickle
import requests
from queue import Queue, Empty
from math import radians, sin, cos, sqrt, atan2
from streamlit_autorefresh import st_autorefresh
import time
import streamlit.components.v1 as components


@st.cache_resource(show_spinner=True)
def load_model():
    
    model_path = st.secrets.get("MODEL_PATH", os.getenv("MODEL_PATH", "")).strip()
    if not model_path:
        for p in ["best.onnx", "best.pt", "models/best.onnx", "models/best.pt"]:
            if os.path.exists(p):
                model_path = p
                break
    if not model_path or not os.path.exists(model_path):
        st.error("Model file not found. Put 'best.onnx' (recommended) or 'best.pt' in repo, or set secrets MODEL_PATH.")
        st.stop()

    model = YOLO(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model.to("cuda")
        try:
            model.model.half()
        except Exception:
            pass
    try:
        model.fuse()
    except Exception:
        pass

    
    try:
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        _ = model.predict(dummy, imgsz=320, conf=0.3, verbose=False)
    except Exception:
        pass
    return model

model = load_model()


disposal_suggestions = {
    "Dry Waste": "Please place this in the dry waste bin (usually blue or gray).",
    "Recyclable": "Please place this in the recyclable waste bin (usually green).",
    "Hazardous Waste": "Please do not place this in normal waste bin. Handle with care and dispose at a hazardous waste collection point (usually red).",
    "Wet Waste": "Please place this in the wet waste bin (usually brown or compost bin)."
}
battery_related_classes = ["Powerbank- (Hazardous Waste)", "Battery (Hazardous Waste)"]

def get_waste_type(label):
    if " (" in label and ")" in label:
        return label.split(" (")[1].rstrip(")")
    return "Unknown"

USER_DATA_FILE = "users.pkl"

def load_users():
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}

def save_users(users_dict):
    with open(USER_DATA_FILE, "wb") as f:
        pickle.dump(users_dict, f)

def get_coordinates(location):
    url = f"https://nominatim.openstreetmap.org/search?q={requests.utils.quote(location)}&format=json"
    try:
        response = requests.get(url, headers={'User-Agent': 'GarbageDetectionApp'})
        if response.status_code == 200:
            data = response.json()
            if data:
                return float(data[0]['lat']), float(data[0]['lon'])
    except Exception:
        pass
    return 0, 0

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def get_nearest_recycling_centers(lat, lon):
    api_key_places = st.secrets.get("GOOGLE_PLACES_API_KEY", "")
    if not api_key_places:
        st.info("GOOGLE_PLACES_API_KEY not set in secrets; Places lookup disabled.")
        return []
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key_places,
        "X-Goog-FieldMask": "places.displayName,places.location,places.formattedAddress"
    }
    payload = {
        "textQuery": "recycling center",
        "locationBias": {"circle": {"center": {"latitude": lat, "longitude": lon}, "radius": 50000}},
        "maxResultCount": 3,
        "rankPreference": "DISTANCE"
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            centers = []
            for place in data.get("places", []):
                name = place["displayName"].get("text", "Unknown Name")
                place_lat = place["location"]["latitude"]; place_lon = place["location"]["longitude"]
                address = place.get("formattedAddress", "Address not available")
                dist = calculate_distance(lat, lon, place_lat, place_lon)
                centers.append({'name': name, 'lat': place_lat, 'lon': place_lon, 'dist': dist, 'address': address})
            if not centers:
                st.info("No places found near your location.")
            return centers
        else:
            try:
                msg = response.json().get('error', {}).get('message', 'Unknown')
            except Exception:
                msg = response.text[:200]
            st.error(f"Google Places API error: {response.status_code}, Message: {msg}")
    except Exception as e:
        st.error(f"Error fetching places: {str(e)}")
    return []

def display_battery_recycling_section(key_suffix=""):
    st.warning("Hazardous battery detected! Please recycle properly.")
    with st.container():
        st.markdown('<div class="location-container">', unsafe_allow_html=True)
        location = st.text_input(
            "Enter your city or zip code for nearest recycling center:",
            placeholder="e.g., New York, NY or 10001",
            key=f"location_input_{key_suffix}"
        )
        if location:
            search_query = f"recycling center near {location}"
            maps_url = f"https://www.google.com/maps/search/?api=1&query={requests.utils.quote(search_query)}"
            st.markdown(f"[Click here to search on Google Maps]({maps_url})")

            lat, lon = get_coordinates(location)
            if lat == 0 and lon == 0:
                st.info("Invalid location. Please enter a valid city or zip code.")
            else:
                centers = get_nearest_recycling_centers(lat, lon)
                if centers:
                    st.write("Nearest Recycling Centers (click to view on map):")
                    col_buttons = st.columns(min(3, len(centers)))
                    for idx, center in enumerate(centers):
                        with col_buttons[idx]:
                            if st.button(f"{center['name']} ({center['dist']:.2f} km) - {center['address']}",
                                         key=f"center_button_{idx}_{key_suffix}"):
                                st.session_state[f"selected_center_{key_suffix}"] = center

                    embed_key = st.secrets.get("GOOGLE_MAPS_EMBED_KEY", "")
                    if f"selected_center_{key_suffix}" in st.session_state:
                        selected = st.session_state[f"selected_center_{key_suffix}"]
                        selected_query = f"{selected['name']} {selected['address']}"
                        embed_src = (
                            f"https://www.google.com/maps/embed/v1/place?key={embed_key}"
                            f"&q={requests.utils.quote(selected_query)}&center={selected['lat']},{selected['lon']}&zoom=15"
                        )
                    else:
                        embed_src = (
                            f"https://www.google.com/maps/embed/v1/search?key={embed_key}"
                            f"&q={requests.utils.quote(search_query)}&center={lat},{lon}&zoom=10"
                        )
                    components.html(
                        f'<iframe width="700" height="500" frameborder="0" style="border:0" '
                        f'referrerpolicy="no-referrer-when-downgrade" src="{embed_src}" allowfullscreen></iframe>',
                        width=700, height=500
                    )
                else:
                    st.info("No nearby recycling centers found. Try the Google Maps link above.")
        st.markdown('</div>', unsafe_allow_html=True)


st.markdown("""
<style>
.stApp {
    padding: 20px;
    font-family: Arial, sans-serif;
    position: relative;
    min-height: 100vh;
}
.stSidebar {
    padding: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.st-expander {
    border-radius: 5px;
    padding: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 5px;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
}
.stButton>button:hover {
    background-color: #45a049;
}
[data-testid="column"] {
    padding: 10px;
    box-sizing: border-box;
}
@media (max-width: 768px) {
    [data-testid="column"] {
        margin-bottom: 20px;
    }
    .row-widget.stHorizontal {
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: center;
        display: flex !important;
    }
    .stVideo, .stImage {
        width: 100% !important;
    }
    .stSidebar {
        padding: 5px;
    }
    .stApp {
        padding: 10px;
    }
    .leaderboard-container .row-widget.stHorizontal [data-testid="column"] {
        flex: 1 1 auto !important;
        min-width: 0 !important;
        max-width: 33.33% !important;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        padding: 5px !important;
        font-size: 0.8em !important;
    }
    .leaderboard-container .row-widget.stHorizontal {
        display: flex !important;
        flex-direction: row !important;
        justify-content: space-between !important;
        align-items: center !important;
        width: 100% !important;
        flex-wrap: nowrap !important;
    }
    .leaderboard-container {
        width: 100% !important;
        overflow-x: auto !important;
    }
    .leaderboard-container [data-testid="stHorizontalBlock"] {
        min-width: 300px !important;
    }
}
.stAlert {
    border-radius: 5px;
    padding: 10px;
}
.stProgress > div > div > div > div {
    background-color: #4CAF50;
}
.login-container, .location-container, .leaderboard-container {
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.user-info {
    font-size: 18px;
    font-weight: bold;
    color: #4CAF50;
}
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
    padding: 10px;
}
.big-title {
    font-size: 1.5em;
    font-weight: bold;
}
.stTextInput > div > div > input {
    border-radius: 4px;
    padding: 8px;
}
.stFileUploader > div {
    border-radius: 4px;
    padding: 8px;
}
.leaderboard-container {
    width: auto;
    max-width: 600px;
    margin: 0 auto;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
}
.leaderboard-header, .leaderboard-row {
    display: flex;
    justify-content: space-between;
    padding: 12px 15px;
}
.leaderboard-header {
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.leaderboard-header span, .leaderboard-row span {
    flex: 1;
    text-align: center;
}
.leaderboard-row span:first-child, .leaderboard-row span:last-child {
    flex: 0.5;
}
.leaderboard-row:hover {
    transition: background-color 0.2s ease;
}
@media (max-width: 768px) {
    .leaderboard-container {
        max-width: 100% !important;
        font-size: 0.9em !important;
    }
    .leaderboard-header, .leaderboard-row {
        padding: 10px 12px;
    }
}

html[theme="light"] {
    .stApp {
        background-color: #f0f2f6;
    }
    .stSidebar {
        background颜色: #ffffff;
    }
    .st-expander {
        border: 1px solid #e0e0e0;
        background-color: #ffffff;
    }
    .stAlert {
        background-color: transparent;
    }
    .login-container, .leaderboard-container {
        border: 1px solid #ddd;
        background-color: #fff;
    }
    .footer {
        background-color: #f0f2f6;
        color: black;
    }
    .stTextInput > div > div > input {
        background-color: #f9f9f9;
        border: 1px solid #ccc;
    }
    .stFileUploader > div {
        background-color: #f9f9f9;
        border: 1px solid #ccc;
    }
    p, div, span, h1, h2, h3, h4, h5, h6, .stMarkdown, [data-testid="stMarkdownContainer"], [data-testid="stExpanderSummary"], [data-testid="stExpanderDetails"], [data-testid="stText"], [data-testid="caption"], .stAlert > div, .stInfo > div, .stWarning > div, .stError > div, .stSuccess > div {
        color: #000000;
    }
    [data-testid="stHorizontalBlock"], [data-testid="column"], .row-widget.stHorizontal, [data-testid="stSidebar"] > div:first-child {
        background-color: transparent;
    }
    .leaderboard-header {
        background-color: #f8f9fa;
        color: #333333;
        border-bottom: 2px solid #dee2e6;
    }
    .leaderboard-row {
        border-bottom: 1px solid #dee2e6;
        color: #555555;
    }
    .leaderboard-row:nth-child(even) {
        background-color: #f8f9fa;
    }
    .leaderboard-row:hover {
        background-color: #e9ecef;
    }
    .leaderboard-row:last-child {
        border-bottom: none;
    }
}

html[theme="dark"] {
    .stApp {
        background-color: #121212;
    }
    .stSidebar {
        background-color: #1f1f1f;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .st-expander {
        border: 1px solid #444444;
        background-color: #1f1f1f;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stAlert {
        background-color: #333333;
    }
    .login-container, .leaderboard-container {
        border: 1px solid #444444;
        background-color: #1f1f1f;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .footer {
        background-color: #121212;
        color: #ffffff;
    }
    .stTextInput > div > div > input {
        background-color: #333333;
        border: 1px solid #555555;
        color: #ffffff;
    }
    .stFileUploader > div {
        background-color: #333333;
        border: 1px solid #555555;
        color: #ffffff;
    }
    p, div, span, h1, h2, h3, h4, h5, h6, .stMarkdown, [data-testid="stMarkdownContainer"], [data-testid="stExpanderSummary"], [data-testid="stExpanderDetails"], [data-testid="stText"], [data-testid="caption"], .stAlert > div, .stInfo > div, .stWarning > div, .stError > div, .stSuccess > div {
        color: #ffffff;
    }
    [data-testid="stHorizontalBlock"], [data-testid="column"], .row-widget.stHorizontal, [data-testid="stSidebar"] > div:first-child {
        background-color: #1f1f1f;
    }
    .leaderboard-header {
        background-color: #2c2c2c;
        color: #ffffff;
        border-bottom: 2px solid #3a3a3a;
    }
    .leaderboard-row {
        border-bottom: 1px solid #3a3a3a;
        color: #dddddd;
    }
    .leaderboard-row:nth-child(even) {
        background-color: #242424;
    }
    .leaderboard-row:hover {
        background-color: #303030;
    }
    .leaderboard-row:last-child {
        border-bottom: none;
    }
}
</style>
""", unsafe_allow_html=True)


st.title("Garbage Detection System :wastebasket:")
st.markdown("**No Login Required, Free Access!** Choose a function to classify garbage. Max file size: 50MB.")

users = load_users()
if "user" not in st.session_state:
    st.session_state.user = None
    st.session_state.points = 0
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "current_detections" not in st.session_state:
    st.session_state.current_detections = []
if "selected_option" not in st.session_state:
    st.session_state.selected_option = "Upload pic/video"
if "points_added" not in st.session_state:
    st.session_state.points_added = False

option = st.sidebar.radio("Function", ("Upload pic/video", "Use Camera"), format_func=lambda x: f"🌟 {x}", key="selected_option")
show_cam_in_result = st.sidebar.checkbox("Show camera results in 'Result' panel (may increase latency)", value=True)


results_container = None
if option == "Upload pic/video":
    results_container = st.expander("📋 Result:", expanded=True)
elif option == "Use Camera" and show_cam_in_result:
    results_container = st.expander("📋 Result:", expanded=True)


if not st.session_state.user:
    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        col_login1, col_login2 = st.columns(2)
        with col_login1:
            username = st.text_input("Username (optional)", placeholder="Enter your username")
        with col_login2:
            password = st.text_input("Password (optional)", type="password", placeholder="Enter your password")
        if st.button("Login/Register"):
            if username and password:
                if username in users:
                    if users[username]["password"] == password:
                        st.session_state.user = username
                        st.session_state.points = users[username]["points"]
                        st.success(f"Welcome back, {username}!")
                        st.rerun()
                    else:
                        st.error("Incorrect password.")
                else:
                    users[username] = {"password": password, "points": 0}
                    save_users(users)
                    st.session_state.user = username
                    st.session_state.points = 0
                    st.success(f"Registered and logged in as {username}!")
                    st.rerun()
            else:
                st.error("Please provide username and password.")
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<p class="user-info">Logged in as: {st.session_state.user} | Points: {st.session_state.points}</p>', unsafe_allow_html=True)
    if st.button("Logout"):
        st.session_state.user = None
        st.session_state.points = 0
        st.session_state.points_added = False
        if "last_uploaded" in st.session_state: del st.session_state.last_uploaded
        if "camera_started" in st.session_state: del st.session_state.camera_started
        st.session_state.current_detections = []
        st.session_state.uploader_key += 1
        st.success("Logged out.")
        st.rerun()

    
    with st.container():
        st.markdown('<div class="leaderboard-container">', unsafe_allow_html=True)
        sorted_users = sorted(users.items(), key=lambda x: x[1]['points'], reverse=True)[:10]
        st.write("Leaderboard (Top 10):")
        leaderboard_html = '<div class="leaderboard-header"><span>Rank</span><span>User</span><span>Points</span></div>'
        for rank, (user, data) in enumerate(sorted_users, 1):
            leaderboard_html += f'<div class="leaderboard-row"><span>{rank}</span><span>{user}</span><span>{data["points"]}</span></div>'
        st.markdown(leaderboard_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def is_battery_related(detections):
    for label, _ in detections:
        if any(b in label for b in battery_related_classes):
            return True
    return False


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detections_queue = Queue(maxsize=1)  
        self._last_annotated = None
        self._frame_idx = 0
        self._last_infer_ts = 0.0
        self._imgsz = 320
        self._conf = 0.35

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self._frame_idx += 1

        
        do_infer = (self._frame_idx % 2 == 0)
        now = time.time()
        if now - self._last_infer_ts < 0.12:
            do_infer = False

        if do_infer:
            try:
                self._last_infer_ts = now
                with torch.no_grad():
                    results = model(img, imgsz=self._imgsz, conf=self._conf, verbose=False)[0]
                annotated = results.plot()

                uniq = set()
                for box in getattr(results, "boxes", []):
                    lbl = model.names[int(box.cls)]
                    uniq.add(lbl)
                    try:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        waste_type = get_waste_type(lbl)
                        suggestion = disposal_suggestions.get(waste_type, "Unknown")
                        cv2.putText(annotated, suggestion, (x1, max(y2 + 20, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    except Exception:
                        pass

                self._last_annotated = annotated
                try:
                    self.detections_queue.put_nowait(sorted(list(uniq)))
                except Exception:
                    try:
                        _ = self.detections_queue.get_nowait()
                        self.detections_queue.put_nowait(sorted(list(uniq)))
                    except Exception:
                        pass
            except Exception:
                self._last_annotated = img

        out = self._last_annotated if self._last_annotated is not None else img
        return av.VideoFrame.from_ndarray(out, format="bgr24")


if option == "Upload pic/video":
    st.subheader("📤 Upload your file here")
    uploaded_files = st.file_uploader("Select Pics (Maximum 10 pics) or Video",
                                      type=["jpg", "jpeg", "png", "mp4"],
                                      accept_multiple_files=True,
                                      help="Multiple Pictures Supported",
                                      key=f"file_uploader_{st.session_state.uploader_key}")

    if uploaded_files:
        names = [f.name for f in uploaded_files]
        if st.session_state.get("last_uploaded") != names:
            st.session_state.last_uploaded = names
            st.session_state.points_added = False

        if len(uploaded_files) > 10:
            st.error("❌ Maximum 10 pics")
        else:
            all_detections = []
            all_unique_types = set()
            has_battery = False
            progress_bar = st.progress(0)
            total_files = len(uploaded_files)
            with st.spinner("🔄 Processing..."):
                try:
                    for idx, uploaded_file in enumerate(uploaded_files, 1):
                        progress_bar.progress(idx / max(total_files, 1))
                        file_type = uploaded_file.type

                        if "image" in file_type:
                            img = np.frombuffer(uploaded_file.read(), np.uint8)
                            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                            with torch.no_grad():
                                results = model(img, imgsz=320, conf=0.35, verbose=False)[0]
                            annotated_img = results.plot()
                            for box in getattr(results, "boxes", []):
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                label = model.names[int(box.cls)]
                                waste_type = get_waste_type(label)
                                suggestion = disposal_suggestions.get(waste_type, "Unknown")
                                cv2.putText(annotated_img, suggestion, (x1, max(y2 + 20, 10)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            st.markdown(f"**File: {uploaded_file.name}**")
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Picture", use_container_width=True)
                            with col2:
                                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Result", use_container_width=True)
                            if getattr(results, "boxes", None):
                                for box in results.boxes:
                                    cls = int(box.cls); conf = float(box.conf)
                                    label = model.names[cls]
                                    waste_type = get_waste_type(label)
                                    all_unique_types.add(waste_type)
                                    all_detections.append((label, conf))
                                    if any(b in label for b in battery_related_classes):
                                        has_battery = True

                        elif "video" in file_type:
                            if len(uploaded_files) > 1:
                                st.error("❌ You could only upload one video at a time")
                                break
                            temp_path = "temp_video.mp4"
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.read())
                            cap = cv2.VideoCapture(temp_path)
                            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            fps = cap.get(cv2.CAP_PROP_FPS) or 25
                            st.write(f"Video Length: {frame_count / max(fps,1):.2f} secs")
                            detections = []
                            unique_types = set()
                            step = max(1, frame_count // 10)
                            for i in range(0, min(frame_count, 10 * step), step):
                                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                                ret, frame_img = cap.read()
                                if ret:
                                    with torch.no_grad():
                                        results = model(frame_img, imgsz=320, conf=0.35, verbose=False)[0]
                                    if getattr(results, "boxes", None):
                                        for box in results.boxes:
                                            cls = int(box.cls)
                                            label = model.names[cls]
                                            waste_type = get_waste_type(label)
                                            unique_types.add(waste_type)
                                            detections.append(label)
                                            if any(b in label for b in battery_related_classes):
                                                has_battery = True
                            cap.release()
                            st.video(temp_path)
                            if results_container:
                                with results_container:
                                    if detections:
                                        unique_detections = sorted(set(detections))
                                        st.markdown('<span class="big-title">Garbage in the video:</span>', unsafe_allow_html=True)
                                        for det in unique_detections:
                                            st.write(f"- {det}")
                                        st.markdown('<span class="big-title">Disposal Advices:</span>', unsafe_allow_html=True)
                                        for waste_type in sorted(unique_types):
                                            suggestion = disposal_suggestions.get(waste_type, "Unknown")
                                            st.write(f"- {waste_type}: {suggestion}")
                                        if has_battery:
                                            display_battery_recycling_section(key_suffix="video")
                                    else:
                                        st.info("No garbage detected")
                            
                            if st.session_state.user and detections and not st.session_state.get("points_added", False):
                                users[st.session_state.user]["points"] = users.get(st.session_state.user, {"points":0}).get("points",0) + len(set(detections))
                                save_users(users)
                                st.session_state.points = users[st.session_state.user]["points"]
                                st.session_state.points_added = True
                                st.success(f"Points updated! You now have {st.session_state.points} points.")
                                st.rerun()
                            continue

                    
                    if st.session_state.user and all_detections and not st.session_state.get("points_added", False):
                        users[st.session_state.user]["points"] = users.get(st.session_state.user, {"points":0}).get("points",0) + len(all_detections)
                        save_users(users)
                        st.session_state.points = users[st.session_state.user]["points"]
                        st.session_state.points_added = True
                        st.success(f"Points updated! You now have {st.session_state.points} points.")
                        st.rerun()

                    if results_container:
                        with results_container:
                            if all_detections:
                                st.markdown('<span class="big-title">Garbage in all images:</span>', unsafe_allow_html=True)
                                for label, conf in all_detections:
                                    st.write(f"- {label} (Confidence: {conf:.2f})")
                                st.markdown('<span class="big-title">Disposal Advices:</span>', unsafe_allow_html=True)
                                for waste_type in sorted(all_unique_types):
                                    suggestion = disposal_suggestions.get(waste_type, "Unknown")
                                    st.write(f"- {waste_type}: {suggestion}")
                                if has_battery or is_battery_related(all_detections):
                                    display_battery_recycling_section(key_suffix="images")
                            else:
                                st.info("No garbage detected")
                except Exception as e:
                    st.error(f"Error: {str(e)}. Please check file format.")


elif option == "Use Camera":
    st.subheader("Camera Detection")
    st.info("Click 'Start' to detect")

    
    rtc_config = {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302", "stun:global.stun.twilio.com:3478?transport=udp"]}
        ]
    }
    turn_url = st.secrets.get("TURN_URL", "")
    turn_user = st.secrets.get("TURN_USERNAME", "")
    turn_cred = st.secrets.get("TURN_CREDENTIAL", "")
    if turn_url and turn_user and turn_cred:
        rtc_config["iceServers"].append({"urls": [turn_url], "username": turn_user, "credential": turn_cred})

    media_constraints = {
        "video": {"width": {"ideal": 640, "max": 640},
                  "height": {"ideal": 480, "max": 480},
                  "frameRate": {"ideal": 20, "max": 20}},
        "audio": False
    }

    ctx = webrtc_streamer(
        key="webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        media_stream_constraints=media_constraints,
        video_processor_factory=VideoProcessor,
        async_processing=True,
        video_html_attrs={"autoPlay": True, "playsinline": True, "muted": True},
    )

    
    if show_cam_in_result and ctx.state.playing:
        st_autorefresh(interval=800, key="cam_result_autorefresh")

    if not ctx.state.playing:
        if "camera_started" in st.session_state: del st.session_state["camera_started"]
        st.session_state.current_detections = []
    else:
        if "camera_started" not in st.session_state:
            st.session_state.camera_started = True

        if ctx.video_processor:
            latest = None
            try:
                while True:
                    latest = ctx.video_processor.detections_queue.get_nowait()
            except Empty:
                pass
            if latest is not None:
                st.session_state.current_detections = latest

            if show_cam_in_result and results_container:
                with results_container:
                    unique_detections = st.session_state.get("current_detections", [])
                    if unique_detections:
                        st.markdown('<span class="big-title">Detected Garbage:</span>', unsafe_allow_html=True)
                        for det in unique_detections:
                            st.write(f"- {det}")
                        unique_types = sorted(set(get_waste_type(det) for det in unique_detections))
                        st.markdown('<span class="big-title">Disposal Advices:</span>', unsafe_allow_html=True)
                        for waste_type in unique_types:
                            suggestion = disposal_suggestions.get(waste_type, "Unknown")
                            st.write(f"- {waste_type}: {suggestion}")
                    else:
                        st.info("No garbage detected")
    

st.markdown('<div class="footer">© 2025 Garbage Detection System | Final Update: 2025-08-12</div>', unsafe_allow_html=True)
