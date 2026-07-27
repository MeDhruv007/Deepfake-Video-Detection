import streamlit as st
import numpy as np
import cv2
import tempfile
import os
from model import Meso4
from datetime import datetime

# Initialize the model once and cache it
@st.cache_resource
def load_model():
    model = Meso4(input_shape=(256, 256, 3))
    if not os.path.exists('weights/meso4_best.weights.h5'):
        st.error("Error: Model weights not found! Please ensure 'weights/meso4_best.weights.h5' exists.")
        st.stop()
    model.load_weights('weights/meso4_best.weights.h5')
    return model

def extract_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None 
        
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    margin_x = int(w * 0.2)
    margin_y = int(h * 0.2)
    
    start_x = max(0, x - margin_x)
    start_y = max(0, y - margin_y)
    end_x = min(image.shape[1], x + w + margin_x)
    end_y = min(image.shape[0], y + h + margin_y)
    
    cropped_face = image[start_y:end_y, start_x:end_x]
    return cropped_face

def process_and_predict(file_bytes, is_video, file_name, model):
    try:
        faces = []
        if is_video:
            # Need a temporary file for OpenCV to read video
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(file_bytes)
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            for _ in range(100):
                ret, frame = cap.read()
                if not ret or len(faces) >= 20:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = extract_face(frame_rgb)
                if face is not None:
                    faces.append(face)
            cap.release()
            os.remove(tfile.name)
        else:
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face = extract_face(image)
            if face is not None:
                faces.append(face)

        if not faces:
            return None, "ERROR: Could not detect any faces in the provided media! MesoNet requires localized faces."
        
        preds = []
        for face in faces:
            face_resized = cv2.resize(face, (256, 256))
            face_normalized = face_resized / 255.0
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            p = model.predict(face_batch, verbose=0)[0][0]
            preds.append(p)
            
        pred = sum(preds) / len(preds)
        confidence = pred if pred > 0.5 else 1 - pred
        label = "FAKE" if pred > 0.5 else "REAL"
        
        return {
            'file_name': file_name,
            'label': label,
            'confidence': confidence,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, None
    except Exception as e:
        return None, f"An error occurred: {str(e)}"

st.set_page_config(page_title="Deepfake Dashboard", page_icon="🧛‍♂️", layout="wide")

# Add some custom CSS for a cleaner monochromatic dashboard look
st.markdown("""
<style>
    /* Dark Monochromatic styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
        max-height: 100vh;
        overflow: hidden;
    }
    div[data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 800;
    }
    .history-card {
        padding: 10px; 
        border-radius: 6px; 
        margin-bottom: 8px; 
        border: 1px solid #333;
        background-color: #111;
        font-size: 0.85em;
        line-height: 1.4;
    }
    .fake-label { color: #ff4b4b; font-weight: bold; }
    .real-label { color: #00c853; font-weight: bold; }
    
    /* Control media sizes */
    video, img {
        max-height: 300px !important;
        width: 100% !important;
        object-fit: scale-down !important;
        border-radius: 8px;
        border: 1px solid #333;
    }
    /* Restrict the height of the right column for scrolling history */
    .history-container {
        max-height: 50vh;
        overflow-y: auto;
        padding-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None

model = load_model()

# --- TOP HEADER ROW ---
head_col1, head_col2 = st.columns([4, 1])
with head_col1:
    st.title("🧛‍♂️ Deepfake Detection Dashboard")
with head_col2:
    st.metric(label="Total Scans", value=len(st.session_state.history))

st.markdown("---")

# --- CONTROL ROW ---
ctrl_col1, ctrl_col2 = st.columns([3, 1])
with ctrl_col1:
    tab1, tab2 = st.tabs(["📁 Upload File", "🔗 Paste URL"])
    with tab1:
        uploaded_file = st.file_uploader("Upload Image or Video", type=['mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'], label_visibility="collapsed")
        video_url = ""
    with tab2:
        video_url = st.text_input("Paste a direct URL to a video or image:", placeholder="https://example.com/video.mp4", label_visibility="collapsed")
with ctrl_col2:
    st.write("") # Spacing 
    st.write("") 
    analyze_pressed = st.button("🔍 Analyze Media", type="primary", use_container_width=True)

st.markdown("---")

# --- FETCH URL MEDIA BEFORE RENDER ---
url_file_bytes = None
url_is_video = False
url_file_name = None
url_error = None

if video_url:
    if 'cached_url' not in st.session_state or st.session_state.cached_url != video_url:
        st.session_state.cached_url = video_url
        st.session_state.cached_bytes = None
        st.session_state.cached_is_video = False
        st.session_state.cached_file_name = None
        st.session_state.cached_error = None
        st.session_state.needs_download = True
    
    if analyze_pressed and st.session_state.get('needs_download', False):
        with st.spinner("Downloading media from URL..."):
            try:
                import yt_dlp
                import os
                file_name_default = video_url.split('/')[-1] if '/' in video_url else 'url_media'
                ydl_opts = {
                    'format': 'best[ext=mp4]/best',
                    'outtmpl': 'temp_download_%(id)s.%(ext)s',
                    'quiet': True,
                    'no_warnings': True
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(video_url, download=True)
                    base_filename = ydl.prepare_filename(info_dict)
                    actual_filename = base_filename
                    if not os.path.exists(actual_filename):
                        base_no_ext = os.path.splitext(base_filename)[0]
                        for ext in ['.mkv', '.mp4', '.webm', '.flv']:
                            if os.path.exists(base_no_ext + ext):
                                actual_filename = base_no_ext + ext
                                break
                                
                    with open(actual_filename, 'rb') as f:
                        file_bytes = f.read()
                    os.remove(actual_filename)
                    
                    st.session_state.cached_bytes = file_bytes
                    st.session_state.cached_is_video = True
                    st.session_state.cached_file_name = info_dict.get('title', file_name_default)
                    st.session_state.needs_download = False
            except Exception as e:
                st.session_state.cached_error = f"Failed to download URL: {str(e)}"
                st.session_state.needs_download = False
                
    url_file_bytes = st.session_state.get('cached_bytes')
    url_is_video = st.session_state.get('cached_is_video', False)
    url_file_name = st.session_state.get('cached_file_name')
    url_error = st.session_state.get('cached_error')

# --- MAIN DASHBOARD AREA ---
dash_col1, dash_col2, dash_col3 = st.columns([1.5, 1.5, 1])

with dash_col2:
    st.markdown("#### Media Preview")
    if uploaded_file is not None:
        is_video = uploaded_file.name.lower().endswith(('.mp4', '.avi', '.mov'))
        file_bytes = uploaded_file.read()
        
        if is_video:
            st.video(file_bytes)
        else:
            st.image(file_bytes)
    elif video_url:
        if url_file_bytes is not None:
            if url_is_video:
                st.video(url_file_bytes)
            else:
                st.image(url_file_bytes)
        elif url_error:
            st.error(url_error)
        else:
            st.info("Preview will be available after you click 'Analyze Media'.")
    else:
        st.info("No media selected.")

with dash_col1:
    st.markdown("#### Scan Results")
    
    if analyze_pressed and (uploaded_file is not None or video_url):
        with st.spinner('Analyzing...'):
            error = None
            if uploaded_file is not None:
                file_name = uploaded_file.name
                # file_bytes and is_video are already processed in the preview section
            else:
                file_name = url_file_name
                file_bytes = url_file_bytes
                is_video = url_is_video
                error = url_error
                
                if not file_bytes and not error:
                    error = "Failed to load media from URL."
            
            if not error:
                result, error = process_and_predict(file_bytes, is_video, file_name, model)
            
            if error:
                st.error(error)
                st.session_state.current_result = None
            else:
                st.session_state.current_result = result
                st.session_state.history.insert(0, result)
    
    res = st.session_state.current_result
    if res:
        if res['label'] == "FAKE":
            st.metric(label="Final Verdict", value="🚨 FAKE")
        else:
            st.metric(label="Final Verdict", value="✅ REAL")
            
        st.metric(label="Confidence Score", value=f"{res['confidence']:.2%}")
        st.caption(f"File: `{res['file_name']}`  \nScanned at: {res['timestamp']}")
    elif not analyze_pressed:
        st.write("Upload media and click Analyze to see results.")

with dash_col3:
    hist_col1, hist_col2 = st.columns([2, 1])
    with hist_col1:
        st.markdown("#### History")
    with hist_col2:
        if st.button("Clear", use_container_width=True):
            st.session_state.history.clear()
            st.session_state.current_result = None
            st.rerun()
            
    if not st.session_state.history:
        st.write("No prior scans.")
    else:
        st.markdown("<div class='history-container'>", unsafe_allow_html=True)
        for item in st.session_state.history:
            label_class = "fake-label" if item['label'] == "FAKE" else "real-label"
            st.markdown(f"""
            <div class='history-card'>
                <b>{item['file_name']}</b><br/>
                <span class='{label_class}'>{item['label']}</span> ({item['confidence']:.1%})<br/>
                <small style='color: #666;'>{item['timestamp']}</small>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

