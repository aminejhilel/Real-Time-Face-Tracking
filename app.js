import {
  FaceLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// ==========================================================================
// Global Application State & DOM Selectors
// ==========================================================================
const video = document.getElementById("webcam");
const canvas = document.getElementById("outputCanvas");
const ctx = canvas.getContext("2d");

const toggleCamBtn = document.getElementById("toggleCamBtn");
const startCamPromptBtn = document.getElementById("startCamPromptBtn");
const captureBtn = document.getElementById("captureBtn");
const videoPlaceholder = document.getElementById("videoPlaceholder");
const hudOverlay = document.getElementById("hudOverlay");

const facesCountEl = document.getElementById("facesCount");
const dominantBadgeText = document.getElementById("dominantBadgeText");

// Control Toggles
const showBoxToggle = document.getElementById("showBoxToggle");
const showLandmarksToggle = document.getElementById("showLandmarksToggle");
const showLabelToggle = document.getElementById("showLabelToggle");

// Hero Emotion Card Elements
const dominantEmojiEl = document.getElementById("dominantEmoji");
const dominantEmotionNameEl = document.getElementById("dominantEmotionName");
const dominantScoreTextEl = document.getElementById("dominantScoreText");
const dominantConfidenceFillEl = document.getElementById("dominantConfidenceFill");
const heroCardEl = document.getElementById("heroEmotionCard");

// Status & FPS
const statusIndicator = document.getElementById("statusIndicator");
const statusText = document.getElementById("statusText");
const fpsCounterEl = document.getElementById("fpsCounter");

// Snapshot Modal Elements
const snapshotModal = document.getElementById("snapshotModal");
const closeModalBtn = document.getElementById("closeModalBtn");
const snapshotImg = document.getElementById("snapshotImg");
const snapshotDetails = document.getElementById("snapshotDetails");
const downloadSnapshotBtn = document.getElementById("downloadSnapshotBtn");

// Emotion Categories & Mapping
const EMOTIONS = [
  { key: "surprise", name: "Surprise", icon: "😲", color: "#00E5FF" },
  { key: "angry",    name: "Angry",    icon: "😡", color: "#FF3B30" },
  { key: "fear",     name: "Fear",     icon: "😨", color: "#AF52DE" },
  { key: "happy",    name: "Happy",    icon: "😊", color: "#FFCC00" },
  { key: "sad",      name: "Sad",      icon: "😢", color: "#5856D6" },
  { key: "neutral",  name: "Neutral",  icon: "😐", color: "#8E8E93" },
  { key: "disgust",  name: "Disgust",  icon: "🤢", color: "#34C759" }
];

let faceLandmarker = null;
let isCameraActive = false;
let animationFrameId = null;
let lastVideoTime = -1;
let frameCount = 0;
let lastFpsCheck = performance.now();
let currentFps = 0;
let telemetryChart = null;

// Timeline history data buffers (last 30 readings)
const MAX_HISTORY = 30;
const historyData = {
  labels: Array(MAX_HISTORY).fill(""),
  happy: Array(MAX_HISTORY).fill(0),
  sad: Array(MAX_HISTORY).fill(0),
  surprise: Array(MAX_HISTORY).fill(0),
  neutral: Array(MAX_HISTORY).fill(0),
  angry: Array(MAX_HISTORY).fill(0),
  fear: Array(MAX_HISTORY).fill(0),
  disgust: Array(MAX_HISTORY).fill(0)
};

// ==========================================================================
// Initialize Application & MediaPipe Models
// ==========================================================================
async function initApp() {
  try {
    updateStatus("Loading AI Models...", "warning");
    
    // Initialize Lucide icons
    if (window.lucide) {
      window.lucide.createIcons();
    }
    
    // Initialize Telemetry Chart
    initChart();
    
    // Resolve WASM files for MediaPipe Vision
    const filesetResolver = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    
    // Load Face Landmarker with Blendshapes enabled for emotion detection
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU"
      },
      outputFaceBlendshapes: true,
      runningMode: "VIDEO",
      numFaces: 2
    });

    updateStatus("System Ready", "active");
    console.log("MediaPipe FaceLandmarker loaded successfully.");
  } catch (error) {
    console.error("Error initializing MediaPipe FaceLandmarker:", error);
    updateStatus("Model Error", "error");
  }
}

function updateStatus(message, state) {
  statusText.textContent = message;
  statusIndicator.className = "status-indicator " + (state === "active" ? "active" : "");
}

// ==========================================================================
// Webcam Stream Control
// ==========================================================================
async function startCamera() {
  if (!faceLandmarker) {
    alert("MediaPipe Face Landmarker AI model is still loading. Please wait a moment...");
    return;
  }
  
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user"
      },
      audio: false
    });

    video.srcObject = stream;
    await new Promise((resolve) => (video.onloadedmetadata = resolve));
    video.play();

    isCameraActive = true;
    videoPlaceholder.style.display = "none";
    hudOverlay.style.display = "flex";
    toggleCamBtn.innerHTML = `<i data-lucide="video-off"></i> Stop Camera`;
    toggleCamBtn.classList.replace("btn-primary", "btn-secondary");
    captureBtn.disabled = false;
    
    if (window.lucide) window.lucide.createIcons();

    // Adjust Canvas dimensions to match video
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    // Start Real-Time Render Loop
    renderLoop();
    updateStatus("Tracking Live", "active");
  } catch (err) {
    console.error("Webcam Access Error:", err);
    alert("Unable to access webcam. Please verify camera permissions in your browser.");
    updateStatus("Camera Denied", "error");
  }
}

function stopCamera() {
  isCameraActive = false;
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
  }

  if (video.srcObject) {
    video.srcObject.getTracks().forEach((track) => track.stop());
    video.srcObject = null;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  videoPlaceholder.style.display = "flex";
  hudOverlay.style.display = "none";
  toggleCamBtn.innerHTML = `<i data-lucide="video"></i> Start Camera`;
  toggleCamBtn.classList.replace("btn-secondary", "btn-primary");
  captureBtn.disabled = true;
  
  if (window.lucide) window.lucide.createIcons();
  
  updateStatus("Camera Stopped", "");
  fpsCounterEl.textContent = "0 FPS";
}

// ==========================================================================
// Main Real-Time Render & Face Tracking Loop
// ==========================================================================
function renderLoop() {
  if (!isCameraActive) return;

  // Calculate Real-Time FPS
  const now = performance.now();
  frameCount++;
  if (now - lastFpsCheck >= 1000) {
    currentFps = Math.round((frameCount * 1000) / (now - lastFpsCheck));
    fpsCounterEl.textContent = `${currentFps} FPS`;
    frameCount = 0;
    lastFpsCheck = now;
  }

  // Detect faces if video frame updated
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    
    const startTimeMs = performance.now();
    const results = faceLandmarker.detectForVideo(video, startTimeMs);

    // Clear Canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results && results.faceLandmarks && results.faceLandmarks.length > 0) {
      facesCountEl.textContent = results.faceLandmarks.length;

      // Process first detected primary face
      for (let i = 0; i < results.faceLandmarks.length; i++) {
        const landmarks = results.faceLandmarks[i];
        const blendshapes = results.faceBlendshapes[i] ? results.faceBlendshapes[i].categories : [];

        // 1. Calculate Face Bounding Box (Green Frame)
        const bbox = computeBoundingBox(landmarks, canvas.width, canvas.height);

        // 2. Compute 7 Emotion Scores from Blendshapes & Landmarks
        const emotionScores = computeEmotionScores(blendshapes);
        const dominant = getDominantEmotion(emotionScores);

        // 3. Render Visual Overlay on Canvas
        if (showBoxToggle.checked) {
          drawGreenBoundingBox(ctx, bbox, dominant.name, dominant.score);
        }

        if (showLandmarksToggle.checked) {
          drawLandmarksMesh(ctx, landmarks, canvas.width, canvas.height);
        }

        if (showLabelToggle.checked && !showBoxToggle.checked) {
          drawEmotionTag(ctx, bbox, dominant.name, dominant.score);
        }

        // Only update UI telemetry based on primary tracked face (face 0)
        if (i === 0) {
          updateEmotionDashboard(emotionScores, dominant);
          updateChartTelemetry(emotionScores);
          dominantBadgeText.textContent = `${dominant.name} (${Math.round(dominant.score * 100)}%)`;
        }
      }
    } else {
      facesCountEl.textContent = "0";
      dominantBadgeText.textContent = "Searching...";
    }
  }

  animationFrameId = requestAnimationFrame(renderLoop);
}

// ==========================================================================
// Face Bounding Box & Landmark Calculation
// ==========================================================================
function computeBoundingBox(landmarks, width, height) {
  let minX = Infinity, minY = Infinity;
  let maxX = -Infinity, maxY = -Infinity;

  landmarks.forEach((pt) => {
    const x = pt.x * width;
    const y = pt.y * height;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  });

  // Expand bounding box slightly for clean framing margin
  const padX = (maxX - minX) * 0.15;
  const padY = (maxY - minY) * 0.20;

  return {
    x: Math.max(0, minX - padX),
    y: Math.max(0, minY - padY),
    width: Math.min(width, (maxX - minX) + (padX * 2)),
    height: Math.min(height, (maxY - minY) + (padY * 2))
  };
}

// Draw Neon Green Frame Bounding Box with Label Tag
function drawGreenBoundingBox(ctx, bbox, dominantEmotion, confidence) {
  const { x, y, width, height } = bbox;
  const greenNeon = "#00FF66";

  ctx.save();

  // Bounding Box Shadow Glow
  ctx.shadowColor = "rgba(0, 255, 102, 0.6)";
  ctx.shadowBlur = 12;

  // Main Bounding Box Rectangle (Green Frame)
  ctx.strokeStyle = greenNeon;
  ctx.lineWidth = 3;
  ctx.strokeRect(x, y, width, height);

  // Corner Accents for Futuristic High-Tech Look
  const cornerLength = Math.min(width, height) * 0.15;
  ctx.lineWidth = 5;
  ctx.strokeStyle = "#FFFFFF";

  // Top-Left Corner
  ctx.beginPath();
  ctx.moveTo(x, y + cornerLength);
  ctx.lineTo(x, y);
  ctx.lineTo(x + cornerLength, y);
  ctx.stroke();

  // Top-Right Corner
  ctx.beginPath();
  ctx.moveTo(x + width - cornerLength, y);
  ctx.lineTo(x + width, y);
  ctx.lineTo(x + width, y + cornerLength);
  ctx.stroke();

  // Bottom-Left Corner
  ctx.beginPath();
  ctx.moveTo(x, y + height - cornerLength);
  ctx.lineTo(x, y + height);
  ctx.lineTo(x + cornerLength, y + height);
  ctx.stroke();

  // Bottom-Right Corner
  ctx.beginPath();
  ctx.moveTo(x + width - cornerLength, y + height);
  ctx.lineTo(x + width, y + height);
  ctx.lineTo(x + width, y + height - cornerLength);
  ctx.stroke();

  ctx.restore();

  // Emotion Tag Badge above Bounding Box
  drawEmotionTag(ctx, bbox, dominantEmotion, confidence);
}

function drawEmotionTag(ctx, bbox, emotionText, confidence) {
  const percentStr = `${Math.round(confidence * 100)}%`;
  const text = `${emotionText.toUpperCase()} ${percentStr}`;

  ctx.save();
  ctx.font = "bold 16px 'Outfit', sans-serif";
  const textWidth = ctx.measureText(text).width;
  const paddingX = 12;
  const badgeHeight = 28;

  const tagX = bbox.x;
  const tagY = Math.max(badgeHeight, bbox.y - 8);

  // Background Badge Pill
  ctx.fillStyle = "rgba(4, 18, 7, 0.9)";
  ctx.shadowColor = "rgba(0,0,0,0.5)";
  ctx.shadowBlur = 8;
  
  ctx.beginPath();
  ctx.roundRect(tagX, tagY - badgeHeight, textWidth + (paddingX * 2), badgeHeight, 6);
  ctx.fill();

  // Left Color Strip
  ctx.fillStyle = "#00FF66";
  ctx.fillRect(tagX, tagY - badgeHeight, 4, badgeHeight);

  // Text Tag
  ctx.fillStyle = "#00FF66";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(text, tagX + paddingX, tagY - (badgeHeight / 2));

  ctx.restore();
}

function drawLandmarksMesh(ctx, landmarks, width, height) {
  ctx.save();
  ctx.fillStyle = "rgba(0, 229, 255, 0.7)";
  landmarks.forEach((pt, index) => {
    // Render key facial anchor points
    if (index % 4 === 0) {
      ctx.beginPath();
      ctx.arc(pt.x * width, pt.y * height, 1.5, 0, 2 * Math.PI);
      ctx.fill();
    }
  });
  ctx.restore();
}

// ==========================================================================
// Real-Time 7-Emotion Inference Engine (MediaPipe Blendshape Analysis)
// ==========================================================================
function computeEmotionScores(blendshapes) {
  const scoresMap = {};
  blendshapes.forEach((b) => {
    scoresMap[b.categoryName] = b.score;
  });

  const getS = (name) => scoresMap[name] || 0;

  // 1. Happy: Smile blendshapes + mouth corner depressor check
  const happyRaw = (getS("mouthSmileLeft") + getS("mouthSmileRight")) / 2 + (getS("cheekRaiserLeft") + getS("cheekRaiserRight")) / 4;

  // 2. Sad: Frown blendshapes + brow inner elevator + mouth roll lower
  const sadRaw = (getS("mouthFrownLeft") + getS("mouthFrownRight")) / 2 + getS("browInnerUp") * 0.6 + getS("mouthRollLower") * 0.4;

  // 3. Surprise: Jaw open + eye wide + brow outer elevator
  const surpriseRaw = getS("jawOpen") * 0.65 + (getS("browOuterUpLeft") + getS("browOuterUpRight")) / 4 + (getS("eyeWideLeft") + getS("eyeWideRight")) / 4;

  // 4. Angry: Brow lowerer + mouth press + nose sneer
  const angryRaw = (getS("browDownLeft") + getS("browDownRight")) / 2 + (getS("mouthPressLeft") + getS("mouthPressRight")) / 4 + getS("noseSneerLeft") * 0.3;

  // 5. Fear: Brow inner up + eye wide + mouth stretch
  const fearRaw = getS("browInnerUp") * 0.5 + (getS("eyeWideLeft") + getS("eyeWideRight")) / 4 + (getS("mouthStretchLeft") + getS("mouthStretchRight")) / 4;

  // 6. Disgust: Nose sneer + mouth upper lip elevator
  const disgustRaw = (getS("noseSneerLeft") + getS("noseSneerRight")) / 2 + (getS("mouthUpperUpLeft") + getS("mouthUpperUpRight")) / 4;

  // 7. Neutral: Remaining facial composure
  const maxExpressionSum = happyRaw + sadRaw + surpriseRaw + angryRaw + fearRaw + disgustRaw;
  const neutralRaw = Math.max(0.05, 1.0 - (maxExpressionSum * 1.4));

  const rawMap = {
    surprise: Math.max(0, surpriseRaw),
    angry:    Math.max(0, angryRaw),
    fear:     Math.max(0, fearRaw),
    happy:    Math.max(0, happyRaw),
    sad:      Math.max(0, sadRaw),
    neutral:  Math.max(0, neutralRaw),
    disgust:  Math.max(0, disgustRaw)
  };

  // Softmax / Normalization so sum of all 7 percentage bars equals 100%
  const total = Object.values(rawMap).reduce((a, b) => a + b, 0) || 1;
  const normalized = {};
  EMOTIONS.forEach((e) => {
    normalized[e.key] = rawMap[e.key] / total;
  });

  return normalized;
}

function getDominantEmotion(scores) {
  let dominantKey = "neutral";
  let maxScore = -1;

  Object.entries(scores).forEach(([key, score]) => {
    if (score > maxScore) {
      maxScore = score;
      dominantKey = key;
    }
  });

  const emoObj = EMOTIONS.find((e) => e.key === dominantKey) || EMOTIONS[5];
  return {
    key: dominantKey,
    name: emoObj.name.toUpperCase(),
    icon: emoObj.icon,
    color: emoObj.color,
    score: maxScore
  };
}

// ==========================================================================
// Dashboard UI Telemetry Updates
// ==========================================================================
function updateEmotionDashboard(scores, dominant) {
  // Update 7 Emotion Percentage Progress Bars
  EMOTIONS.forEach((emo) => {
    const scoreVal = scores[emo.key] || 0;
    const percent = Math.round(scoreVal * 100);

    const barFill = document.getElementById(`bar-${emo.key}`);
    const valText = document.getElementById(`val-${emo.key}`);

    if (barFill) barFill.style.width = `${percent}%`;
    if (valText) valText.textContent = `${percent}%`;
  });

  // Update Hero Dominant Emotion Card
  dominantEmojiEl.textContent = dominant.icon;
  dominantEmotionNameEl.textContent = dominant.name;
  dominantEmotionNameEl.style.color = dominant.color;
  dominantScoreTextEl.textContent = `${(dominant.score * 100).toFixed(1)}% Confidence`;
  dominantConfidenceFillEl.style.width = `${Math.round(dominant.score * 100)}%`;
  dominantConfidenceFillEl.style.background = dominant.color;
  heroCardEl.style.borderLeftColor = dominant.color;
}

// ==========================================================================
// Telemetry Chart (Chart.js Line Chart)
// ==========================================================================
function initChart() {
  const chartCanvas = document.getElementById("telemetryChart");
  if (!chartCanvas) return;

  const ctxChart = chartCanvas.getContext("2d");

  telemetryChart = new Chart(ctxChart, {
    type: "line",
    data: {
      labels: historyData.labels,
      datasets: [
        { label: "Happy", data: historyData.happy, borderColor: "#FFCC00", borderWidth: 2, tension: 0.3, pointRadius: 0 },
        { label: "Sad", data: historyData.sad, borderColor: "#5856D6", borderWidth: 2, tension: 0.3, pointRadius: 0 },
        { label: "Surprise", data: historyData.surprise, borderColor: "#00E5FF", borderWidth: 2, tension: 0.3, pointRadius: 0 },
        { label: "Neutral", data: historyData.neutral, borderColor: "#8E8E93", borderWidth: 2, tension: 0.3, pointRadius: 0 },
        { label: "Angry", data: historyData.angry, borderColor: "#FF3B30", borderWidth: 2, tension: 0.3, pointRadius: 0 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { display: true, position: "top", labels: { color: "#94A3B8", boxWidth: 10, font: { size: 10 } } }
      },
      scales: {
        x: { display: false },
        y: { min: 0, max: 100, ticks: { color: "#64748B", font: { size: 9 } }, grid: { color: "rgba(255, 255, 255, 0.05)" } }
      }
    }
  });
}

function updateChartTelemetry(scores) {
  if (!telemetryChart) return;

  EMOTIONS.forEach((emo) => {
    if (historyData[emo.key]) {
      historyData[emo.key].shift();
      historyData[emo.key].push(Math.round((scores[emo.key] || 0) * 100));
    }
  });

  telemetryChart.update("none");
}

// ==========================================================================
// Snapshot Capture & Modal
// ==========================================================================
function captureSnapshot() {
  if (!isCameraActive) return;

  // Create temporary canvas combining webcam video frame and overlay canvas
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = canvas.width;
  tempCanvas.height = canvas.height;
  const tempCtx = tempCanvas.getContext("2d");

  // Mirror draw video frame
  tempCtx.save();
  tempCtx.scale(-1, 1);
  tempCtx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
  tempCtx.restore();

  // Draw canvas overlay
  tempCtx.drawImage(canvas, 0, 0);

  const dataUrl = tempCanvas.toDataURL("image/png");
  snapshotImg.src = dataUrl;
  downloadSnapshotBtn.href = dataUrl;

  const currentDominant = dominantEmotionNameEl.textContent;
  const currentConfidence = dominantScoreTextEl.textContent;
  snapshotDetails.textContent = `Tracked Emotion: ${currentDominant} (${currentConfidence}) | Captured at ${new Date().toLocaleTimeString()}`;

  snapshotModal.style.display = "flex";
}

// ==========================================================================
// Event Listeners
// ==========================================================================
toggleCamBtn.addEventListener("click", () => {
  if (isCameraActive) stopCamera();
  else startCamera();
});

startCamPromptBtn.addEventListener("click", startCamera);

captureBtn.addEventListener("click", captureSnapshot);

closeModalBtn.addEventListener("click", () => {
  snapshotModal.style.display = "none";
});

snapshotModal.addEventListener("click", (e) => {
  if (e.target === snapshotModal) snapshotModal.style.display = "none";
});

// Run App Initialization on Load
window.addEventListener("DOMContentLoaded", initApp);
