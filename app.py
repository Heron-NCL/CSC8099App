import sys
sys.path.insert(0, '.')
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av
import torch
from datetime import datetime
import pickle
import os
import folium
import streamlit.components.v1 as components
import requests
from queue import Queue, Empty
import time
from math import radians, sin, cos, sqrt, atan2

@st.cache_resource
def load_model():
    model_path = "best.pt"
    model = YOLO(model_path)
    if torch.cuda.is_available():
        model.to("cuda")
    return model

model = load_model()
model.fuse()
model.eval()

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
        with open(USER_DATA_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_users(users):
    with open(USER_DATA_FILE, "wb") as f:
        pickle.dump(users, f)

def get_coordinates(location):
    url = f"https://nominatim.openstreetmap.org/search?q={requests.utils.quote(location)}&format=json"
    try:
        response = requests.get(url, headers={'User-Agent': 'GarbageDetectionApp'})
        if response.status_code == 200:
            data = response.json()
            if data:
                return float(data[0]['lat']), float(data[0]['lon'])
    except:
        pass
    return 0, 0  # Default to global if fails

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def get_nearest_recycling_centers(lat, lon):
    api_key = "AIzaSyC-Jo63ACxxufOlHxlBwhYNblKpJwe-ztk"
    url = f"https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.displayName,places.location,places.formattedAddress"
    }
    payload = {
        "textQuery": "recycling center",
        "locationBias": {
            "circle": {
                "center": {"latitude": lat, "longitude": lon},
                "radius": 50000  # 50km
            }
        },
        "maxResultCount": 3,
        "rankPreference": "DISTANCE"
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if "places" in data:
                centers = []
                for place in data["places"]:
                    name = place["displayName"].get("text", "Unknown Name")
                    place_lat = place["location"]["latitude"]
                    place_lon = place["location"]["longitude"]
                    address = place.get("formattedAddress", "Address not available")
                    dist = calculate_distance(lat, lon, place_lat, place_lon)
                    centers.append({'name': name, 'lat': place_lat, 'lon': place_lon, 'dist': dist, 'address': address})
                return centers
            else:
                st.error("Google Places API error: No places found")
        else:
            st.error(f"Google Places API error: {response.status_code}, Message: {response.json().get('error', {}).get('message', 'Unknown')}")
    except Exception as e:
        st.error(f"Error fetching places: {str(e)}")
    return []

def display_battery_recycling_section(key_suffix=""):
    st.warning("Hazardous battery detected! Please recycle properly.")
    with st.container():
        st.markdown('<div class="location-container">', unsafe_allow_html=True)
        location = st.text_input("Enter your city or zip code for nearest recycling center:", placeholder="e.g., New York, NY or 10001", key=f"location_input_{key_suffix}")
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
                            if st.button(f"{center['name']} ({center['dist']:.2f} km) - {center['address']}", key=f"center_button_{idx}_{key_suffix}"):
                                st.session_state[f"selected_center_{key_suffix}"] = center

                    api_key = "AIzaSyC-Jo63ACxxufOlHxlBwhYNblKpJwe-ztk"
                    if f"selected_center_{key_suffix}" in st.session_state:
                        selected = st.session_state[f"selected_center_{key_suffix}"]
                        selected_query = f"{selected['name']} {selected['address']}"
                        embed_src = f"https://www.google.com/maps/embed/v1/place?key={api_key}&q={requests.utils.quote(selected_query)}&center={selected['lat']},{selected['lon']}&zoom=15"
                    else:
                        embed_src = f"https://www.google.com/maps/embed/v1/search?key={api_key}&q={requests.utils.quote(search_query)}&center={lat},{lon}&zoom=10"

                    components.html(
                        f'<iframe width="700" height="500" frameborder="0" style="border:0" referrerpolicy="no-referrer-when-downgrade" src="{embed_src}" allowfullscreen></iframe>',
                        width=700,
                        height=500
                    )
                else:
                    st.info("No nearby recycling centers found. Try the Google Maps link above.")
                    api_key = "AIzaSyC-Jo63ACxxufOlHxlBwhYNblKpJwe-ztk"
                    embed_src = f"https://www.google.com/maps/embed/v1/search?key={api_key}&q={requests.utils.quote(search_query)}&center={lat},{lon}&zoom=10"
                    components.html(
                        f'<iframe width="700" height="500" frameborder="0" style="border:0" referrerpolicy="no-referrer-when-downgrade" src="{embed_src}" allowfullscreen></iframe>',
                        width=700,
                        height=500
                    )
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
        min-width: 300px !important;  /* Ensure minimum width to prevent squeezing */
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
        background-color: #ffffff;
    }
    .st-expander {
        border: 1px solid #e0e0e0;
        background-color: #ffffff;
    }
    .stAlert {
        background-color: transparent;
    }
    .login-container, .location-container, .leaderboard-container {
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
    .login-container, .location-container, .leaderboard-container {
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
                        # st.rerun() removed to reduce rerun storms during camera streaming
                    else:
                        st.error("Incorrect password.")
                else:
                    users[username] = {"password": password, "points": 0}
                    save_users(users)
                    st.session_state.user = username
                    st.session_state.points = 0
                    st.success(f"Registered and logged in as {username}!")
                    # st.rerun() removed to reduce rerun storms during camera streaming
            else:
                st.error("Please provide username and password.")
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<p class="user-info">Logged in as: {st.session_state.user} | Points: {st.session_state.points}</p>', unsafe_allow_html=True)
    if st.button("Logout"):
        st.session_state.user = None
        st.session_state.points = 0
        if "points_added" in st.session_state:
            del st.session_state.points_added
        if "last_uploaded" in st.session_state:
            del st.session_state.last_uploaded
        if "camera_started" in st.session_state:
            del st.session_state.camera_started
        if "current_detections" in st.session_state:
            del st.session_state.current_detections
        st.session_state.uploader_key += 1
        st.success("Logged out.")
        # st.rerun() removed to reduce rerun storms during camera streaming
    # Display leaderboard
    with st.container():
        st.markdown('<div class="leaderboard-container">', unsafe_allow_html=True)
        sorted_users = sorted(users.items(), key=lambda x: x[1]['points'], reverse=True)[:10]
        st.write("Leaderboard (Top 10):")
        leaderboard_html = '<div class="leaderboard-header"><span>Rank</span><span>User</span><span>Points</span></div>'
        for rank, (user, data) in enumerate(sorted_users, 1):
            leaderboard_html += f'<div class="leaderboard-row"><span>{rank}</span><span>{user}</span><span>{data["points"]}</span></div>'
        st.markdown(leaderboard_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

option = st.sidebar.radio("Function", ("Upload pic/video", "Use Camera"), format_func=lambda x: f"🌟 {x}", key="selected_option")

results_container = st.expander("📋 Result:", expanded=True)

def is_battery_related(detections):
    for label, _ in detections:
        if any(b in label for b in battery_related_classes):
            return True
    return False


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        # Single-slot queue to publish latest unique detections to the UI
        self.detections_queue = Queue(maxsize=1)
        # Cache last annotated frame to display even when skipping inference
        self._last_annotated = None
        self._last_detections = []
        self._frame_idx = 0
        self._last_infer_ts = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # ---- Inference throttling ----
        # Process at most ~8 inferences/sec and only every 2 frames
        self._frame_idx += 1
        do_infer = (self._frame_idx % 2 == 0)

        import time
        now = time.time()
        if now - self._last_infer_ts < 0.12:  # ~8 FPS cap
            do_infer = False

        if do_infer:
            self._last_infer_ts = now
            with torch.no_grad():
                # Smaller imgsz speeds up CPU inference; tweakable in sidebar if needed
                results = model(img, imgsz=320, conf=0.35)[0]

            annotated = results.plot()

            # Build a compact set of unique detection labels for UI
            uniq = set()
            for box in results.boxes:
                lbl = model.names[int(box.cls)]
                uniq.add(lbl)

                # Hint text overlay for disposal guidance (cheap to keep)
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    waste_type = get_waste_type(lbl)
                    suggestion = disposal_suggestions.get(waste_type, "Unknown")
                    cv2.putText(annotated, suggestion, (x1, max(y2 + 20, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception:
                    pass

            self._last_annotated = annotated
            self._last_detections = sorted(list(uniq))

            # Non-blocking publish to queue (drop if UI hasn't consumed the previous one)
            try:
                self.detections_queue.put_nowait(self._last_detections)
            except Exception:
                pass

        # Return last annotated frame (or the raw image if none yet)
        out = self._last_annotated if self._last_annotated is not None else img
        return av.VideoFrame.from_ndarray(out, format="bgr24")
