"""
=============================================================================
Real-Time Face Tracking & Emotion Analyzer - Streamlit Dashboard UI
=============================================================================
Run using:
    streamlit run app.py
"""

import cv2
import time
import numpy as np
import streamlit as st
from emotion_analyzer import EmotionAnalyzer

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Real-Time Face Tracking & Emotion Analyzer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for Streamlit Glassmorphism & Neon Theme
st.markdown("""
    <style>
    .main {
        background-color: #090d16;
        color: #f1f5f9;
    }
    .stApp {
        background: radial-gradient(circle at 50% 50%, rgba(0, 255, 102, 0.03) 0%, transparent 70%);
    }
    .emotion-card {
        background: rgba(18, 25, 41, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 5px solid #00FF66;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .dominant-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #00FF66;
        margin-bottom: 0;
    }
    </style>
""", unsafe_allow_html=True)

# 7 Emotion definitions with emojis and hex colors
EMOTIONS_INFO = {
    'surprise': {'name': 'Surprise', 'icon': '😲', 'color': '#00E5FF'},
    'angry':    {'name': 'Angry',    'icon': '😡', 'color': '#FF3B30'},
    'fear':     {'name': 'Fear',     'icon': '😨', 'color': '#AF52DE'},
    'happy':    {'name': 'Happy',    'icon': '😊', 'color': '#FFCC00'},
    'sad':      {'name': 'Sad',      'icon': '😢', 'color': '#5856D6'},
    'neutral':  {'name': 'Neutral',  'icon': '😐', 'color': '#8E8E93'},
    'disgust':  {'name': 'Disgust',  'icon': '🤢', 'color': '#34C759'}
}

@st.cache_resource
def load_analyzer():
    """Cache the EmotionAnalyzer object so model isn't reloaded on every frame rerun."""
    return EmotionAnalyzer()

def main():
    st.title("🎭 JHILEL FaceTrack AI")
    st.caption("Powered by OpenCV, FER Deep Neural Networks, and Streamlit Dashboard")

    # Initialize Engine
    analyzer = load_analyzer()

    # Sidebar Options
    st.sidebar.header("⚙️ Camera & Model Controls")
    camera_index = st.sidebar.number_input("Select Camera Index", value=0, step=1)
    run_webcam = st.sidebar.checkbox("Start Live Webcam Stream", value=False)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📌 Tracked Emotions")
    for k, v in EMOTIONS_INFO.items():
        st.sidebar.markdown(f"{v['icon']} **{v['name']}**")

    # Layout: Two Columns (Left: Video Feed | Right: Emotion Telemetry Dashboard)
    col_video, col_telemetry = st.columns([1.3, 1])

    with col_video:
        st.subheader("📹 Live Bounding Box Video Feed")
        frame_placeholder = st.empty()
        fps_placeholder = st.empty()

    with col_telemetry:
        st.subheader("📊 Emotion Percentage Analytics")
        
        # Hero Dominant Emotion Metric Card Container
        hero_placeholder = st.empty()
        st.markdown("---")
        st.subheader("7-Emotion Live Score Bars")
        
        # Placeholders for the 7 emotion progress bars
        progress_placeholders = {}
        for emo_key in ['surprise', 'angry', 'fear', 'happy', 'sad', 'neutral', 'disgust']:
            info = EMOTIONS_INFO[emo_key]
            st.write(f"{info['icon']} **{info['name']}**")
            progress_placeholders[emo_key] = {
                'bar': st.progress(0),
                'text': st.empty()
            }

    # Webcam Stream Processing Loop
    if run_webcam:
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            st.error(f"Error: Could not open camera device at index {camera_index}.")
            return

        prev_time = time.time()

        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab video frame.")
                break

            # Process frame with face detection and emotion recognition
            processed_frame, results = analyzer.detect_faces_and_emotions(frame)

            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            fps_placeholder.text(f"Performance: {fps:.1f} FPS")

            # Convert BGR (OpenCV default) to RGB for Streamlit rendering
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Update Telemetry UI if face is detected
            if results and len(results) > 0:
                primary = results[0]
                emotions = primary['emotions']
                dominant = primary['dominant_emotion']
                dom_score = primary['dominant_score']

                dom_info = EMOTIONS_INFO.get(dominant, EMOTIONS_INFO['neutral'])

                # Update Hero Card
                hero_placeholder.markdown(f"""
                    <div class="emotion-card">
                        <span style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase;">Dominant Emotion</span>
                        <div class="dominant-title">{dom_info['icon']} {dom_info['name'].upper()}</div>
                        <span style="font-size: 1.1rem; font-weight: 600; color: #00FF66;">
                            {int(dom_score * 100)}% Confidence
                        </span>
                    </div>
                """, unsafe_allow_html=True)

                # Update all 7 emotion percentage score bars dynamically
                for emo_key in ['surprise', 'angry', 'fear', 'happy', 'sad', 'neutral', 'disgust']:
                    val = float(emotions.get(emo_key, 0.0))
                    val = max(0.0, min(1.0, val)) # Clamp between 0 and 1
                    
                    progress_placeholders[emo_key]['bar'].progress(int(val * 100))
                    progress_placeholders[emo_key]['text'].caption(f"Score: {int(val * 100)}%")
            else:
                hero_placeholder.info("Searching for faces in video stream...")

        cap.release()
    else:
        frame_placeholder.info("Check 'Start Live Webcam Stream' in the sidebar to begin tracking.")

if __name__ == '__main__':
    main()
