***********************************************
# Real-Time Face Tracking & Emotion Analyzer

A dual-architecture **Real-Time Face Tracking & Emotion Analyzer** platform. This project provides two complete implementations:
1. **High-Performance Web Dashboard:** Runs entirely in the browser using HTML5 Canvas and MediaPipe Vision (zero backend latency).
2. **Python OpenCV + Streamlit App:** A robust Python backend leveraging OpenCV Haar Cascades and FER (Facial Expression Recognition) deep learning models.

Both versions feature real-time webcam access, neon green face bounding boxes, dominant emotion badges, and a dynamic 7-emotion percentage dashboard (Surprise, Angry, Fear, Happy, Sad, Neutral, Disgust).

---
*******************************************
## 🎨 Option 1: Web Dashboard (Recommended)
***********************--------------------

The Web Dashboard uses modern Glassmorphism UI, Recharts/Chart.js telemetry, and client-side AI inference via MediaPipe. It is the fastest and most responsive option, running directly on your GPU/CPU via WebAssembly.

### Prerequisites
- Node.js (v18+) or any local web server (e.g., Python `http.server`).

### Quick Start
1. Open a terminal in the root directory.
2. Run the application using `npx`:
   ```bash
   npx serve . -l 3000
   ```
   *(Alternatively, run `python -m http.server 3000`)*
3. Open your browser and navigate to `http://localhost:3000`.
4. Click **Start Camera** and grant webcam permissions to begin real-time emotion tracking!

---

## 🐍 Option 2: Python Streamlit Application

The Python implementation is perfect if you want to extend the computer vision pipeline with custom PyTorch/TensorFlow models or perform server-side processing.

### Prerequisites
- Python 3.9 - 3.14
- A working webcam

### Setup Instructions

1. **Navigate to the Python App Directory:**
2. *******************
   ```bash
   cd python_app
   ```

3. **Create a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```
*****************************
4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This will install `opencv-python`, `fer`, `streamlit`, and required data science libraries.*

5. **Run the Streamlit Dashboard:**
   ```bash
   streamlit run app.py
   ```

6. **Usage:**
   - The Streamlit interface will open in your default browser.
   - In the left sidebar, check **"Start Live Webcam Stream"**.
   - Ensure the correct Camera Index (usually `0`) is selected.

---

## 🛠️ Tech Stack Details

- **Web Version:** HTML5, CSS3, JavaScript (ES6+), MediaPipe Tasks Vision (`@mediapipe/tasks-vision`), Chart.js, Lucide Icons.
- **Python Version:** Python, OpenCV (`cv2`), FER (Facial Expression Recognition), Streamlit.

**************************
## 📝 Features Checklist
- [x] Access system webcam stream in real time.
- [x] Detect and track human faces in the video frame.
- [x] Draw bounding box (neon green frame) around the detected face.
- [x] Display dominant emotion text label on bounding box.
- [x] Real-time emotion recognition on cropped face region / blendshapes.
- [x] Live percentage bars for 7 emotion categories.
- [x] Emotion history timeline chart (Web version).
- [x] Snapshot capture utility (Web version).
