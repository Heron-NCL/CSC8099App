
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

# -------------------- Model loading --------------------
@st.cache_resource
def load_model():
    # Use a relative path so it works on Streamlit Community Cloud
    model_path = os.getenv("MODEL_PATH", "best.pt")
    model = YOLO(model_path)
    if torch.cuda.is_available():
        model.to("cuda")
    # Small fusions for speed
    try:
        model.fuse()
    except Exception:
        pass
    model.eval()
    return model

model = load_model()

# -------------------- App Data & Helpers --------------------
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
    except Exception:
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
    # NOTE: For production, do NOT hardcode API keys. Use secrets manager / env var.
    api_key = os.getenv("GOOGLE_MAPS_EMBED_KEY", "")
    url = f"https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": os.getenv("GOOGLE_PLACES_API_KEY", ""),
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
        response = requests.post(url, json=payload, headers=headers, timeout=10)
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
                st.info("No places found near your location.")
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

                    embed_key = os.getenv("GOOGLE_MAPS_EMBED_KEY", "")
                    if f"selected_center_{key_suffix}" in st.session_state:
                        selected = st.session_state[f"selected_center_{key_suffix}"]
                        selected_query = f"{selected['name']} {selected['address']}"
                        embed_src = f"https://www.google.com/maps/embed/v1/place?key={embed_key}&q={requests.utils.quote(selected_query)}&center={selected['lat']},{selected['lon']}&zoom=15"
                    else:
                        embed_src = f"https://www.google.com/maps/embed/v1/search?key={embed_key}&q={requests.utils.quote(search_query)}&center={lat},{lon}&zoom=10"

                    components.html(
                        f'<iframe width="700" height="500" frameborder="0" style="border:0" referrerpolicy="no-referrer-when-downgrade" src="{embed_src}" allowfullscreen></iframe>',
                        width=700,
                        height=500
                    )
                else:
                    st.info("No nearby recycling centers found. Try the Google Maps link above.")
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Styles --------------------
st.markdown("""
<style>
.footer { text-align:center; padding:10px; }
.big-title { font-size: 1.1rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# -------------------- UI --------------------
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

# (Optional) Login UI preserved
if not st.session_state.user:
    with st.container():
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
                        st.experimental_rerun()
                    else:
                        st.error("Incorrect password.")
                else:
                    users[username] = {"password": password, "points": 0}
                    save_users(users)
                    st.session_state.user = username
                    st.session_state.points = 0
                    st.success(f"Registered and logged in as {username}!")
                    st.experimental_rerun()
            else:
                st.error("Please provide username and password.")
else:
    st.markdown(f'**Logged in as:** {st.session_state.user}  |  **Points:** {st.session_state.points}')
    if st.button("Logout"):
        st.session_state.clear()
        st.success("Logged out.")
        st.experimental_rerun()
    # Leaderboard
    with st.container():
        sorted_users = sorted(users.items(), key=lambda x: x[1]['points'], reverse=True)[:10]
        st.write("Leaderboard (Top 10):")
        for rank, (user, data) in enumerate(sorted_users, 1):
            st.write(f"{rank}. {user} - {data['points']}")

option = st.sidebar.radio("Function", ("Upload pic/video", "Use Camera"), format_func=lambda x: f"🌟 {x}", key="selected_option")

results_container = st.expander("📋 Result:", expanded=True)

def is_battery_related(detections):
    for label, _ in detections:
        if any(b in label for b in battery_related_classes):
            return True
    return False

# -------------------- Real-time Video Processor --------------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detections_queue = Queue(maxsize=1)  # latest-only
        self._last_annotated = None
        self._frame_idx = 0
        self._last_infer_ts = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self._frame_idx += 1

        # Throttle: every 2 frames + <= ~8 infers/sec
        do_infer = (self._frame_idx % 2 == 0)
        now = time.time()
        if now - self._last_infer_ts < 0.12:
            do_infer = False

        if do_infer:
            try:
                self._last_infer_ts = now
                with torch.no_grad():
                    results = model(img, imgsz=320, conf=0.35)[0]
                annotated = results.plot()

                uniq = set()
                for box in results.boxes:
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
                    pass
            except Exception:
                # In case of error, keep last frame to avoid breaking stream
                self._last_annotated = img

        out = self._last_annotated if self._last_annotated is not None else img
        return av.VideoFrame.from_ndarray(out, format="bgr24")

# -------------------- Upload flow --------------------
if option == "Upload pic/video":
    st.subheader("📤 Upload your file here")
    uploaded_files = st.file_uploader("Select Pics (Maximum 10 pics) or Video",
                                      type=["jpg", "jpeg", "png", "mp4"],
                                      accept_multiple_files=True,
                                      help="Multiple Pictures Supported",
                                      key=f"file_uploader_{st.session_state.uploader_key}")
    if uploaded_files:
        names = [f.name for f in uploaded_files]
        if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != names:
            st.session_state.pop("points_added", None)
            st.session_state.last_uploaded = names

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
                        progress_bar.progress(idx / total_files)
                        file_type = uploaded_file.type
                        if "image" in file_type:
                            img = np.frombuffer(uploaded_file.read(), np.uint8)
                            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                            with torch.no_grad():
                                results = model(img, imgsz=320, conf=0.35)[0]
                            annotated_img = results.plot()
                            st.markdown(f"**File: {uploaded_file.name}**")
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Picture", use_column_width=True)
                            with col2:
                                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Result", use_column_width=True)
                            if results.boxes:
                                for box in results.boxes:
                                    cls = int(box.cls)
                                    conf = float(box.conf)
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
                                ret, frame = cap.read()
                                if ret:
                                    with torch.no_grad():
                                        results = model(frame, imgsz=320, conf=0.35)[0]
                                    if results.boxes:
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
                            with results_container:
                                if detections:
                                    unique_detections = set(detections)
                                    st.markdown('<span class="big-title">Garbage in the video:</span>', unsafe_allow_html=True)
                                    for det in unique_detections:
                                        st.write(f"- {det}")
                                    st.markdown('<span class="big-title">Disposal Advices:</span>', unsafe_allow_html=True)
                                    for waste_type in unique_types:
                                        suggestion = disposal_suggestions.get(waste_type, "Unknown")
                                        st.write(f"- {waste_type}: {suggestion}")
                                    if has_battery:
                                        display_battery_recycling_section(key_suffix="video")
                                    if st.session_state.user and "points_added" not in st.session_state:
                                        st.session_state.points_added = True
                                        added_points = len(unique_detections)
                                        users[st.session_state.user]["points"] += added_points
                                        st.session_state.points = users[st.session_state.user]["points"]
                                        save_users(users)
                                        st.success(f"Points updated! You now have {st.session_state.points} points.")
                                else:
                                    st.info("No garbage detected")
                    if st.session_state.user and all_detections and "points_added" not in st.session_state:
                        st.session_state.points_added = True
                        users[st.session_state.user]["points"] += len(all_detections)
                        st.session_state.points = users[st.session_state.user]["points"]
                        save_users(users)
                        st.success(f"Points updated! You now have {st.session_state.points} points.")
                    with results_container:
                        if all_detections:
                            st.markdown('<span class="big-title">Garbage in all images:</span>', unsafe_allow_html=True)
                            for label, conf in all_detections:
                                st.write(f"- {label} (Confidence: {conf:.2f})")
                            st.markdown('<span class="big-title">Disposal Advices:</span>', unsafe_allow_html=True)
                            for waste_type in all_unique_types:
                                suggestion = disposal_suggestions.get(waste_type, "Unknown")
                                st.write(f"- {waste_type}: {suggestion}")
                            if has_battery or is_battery_related(all_detections):
                                display_battery_recycling_section(key_suffix="images")
                        else:
                            st.info("No garbage detected")
                except Exception as e:
                    st.error(f"Error: {str(e)}. Please check file format.")

# -------------------- Camera flow --------------------
elif option == "Use Camera":
    st.subheader("Camera Detection")
    st.info("Click 'Start' to detect")

    # ICE servers: Google STUN + hook for your TURN
    rtc_config = {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            # Uncomment and set your TURN if you are behind restricted networks:
            # {"urls": ["turn:turn.yourdomain.com:3478"], "username": "user", "credential": "pass"}
        ]
    }

    ctx = webrtc_streamer(
        key="webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_processor_factory=VideoProcessor,
        async_processing=True,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 15, "max": 15},
            },
            "audio": False,
        },
        video_html_attrs={"style": {"width": "100%"}},
    )

    status_placeholder = st.empty()
    if not ctx.state.playing:
        st.session_state.pop("camera_started", None)
        st.session_state.pop("points_added", None)
        st.session_state.current_detections = []
        status_placeholder.warning("🟡 Click Start to enable camera")
    else:
        status_placeholder.info("🔴 Camera connected, running inference...")
        if "camera_started" not in st.session_state:
            st.session_state.camera_started = True
        if ctx.video_processor:
            try:
                unique_detections = ctx.video_processor.detections_queue.get(timeout=0.1)
                st.session_state.current_detections = unique_detections
            except Empty:
                unique_detections = st.session_state.current_detections
            with results_container:
                if unique_detections:
                    st.markdown('<span class="big-title">Detected Garbage:</span>', unsafe_allow_html=True)
                    for det in sorted(unique_detections):
                        st.write(f"- {det}")
                    unique_types = set(get_waste_type(det) for det in unique_detections)
                    st.markdown('<span class="big-title">Disposal Advices:</span>', unsafe_allow_html=True)
                    for waste_type in sorted(unique_types):
                        suggestion = disposal_suggestions.get(waste_type, "Unknown")
                        st.write(f"- {waste_type}: {suggestion}")
                    has_battery = any(any(b in det for b in battery_related_classes) for det in unique_detections)
                    if has_battery:
                        display_battery_recycling_section(key_suffix="camera")
                else:
                    st.info("No garbage detected")
            if st.session_state.user and unique_detections and "points_added" not in st.session_state:
                st.session_state.points_added = True
                users[st.session_state.user]["points"] += len(unique_detections)
                st.session_state.points = users[st.session_state.user]["points"]
                save_users(users)
                st.success(f"Points updated! You now have {st.session_state.points} points.")

st.markdown('<div class="footer">© 2025 Garbage Detection System</div>', unsafe_allow_html=True)
