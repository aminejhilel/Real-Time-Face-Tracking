"""
=============================================================================
Real-Time Face Tracking & Emotion Analyzer - Core OpenCV/FER Engine Module
=============================================================================
"""

import cv2
import numpy as np
from fer import FER

class EmotionAnalyzer:
    """
    Modular Computer Vision pipeline for face detection and real-time 7-emotion recognition.
    """
    def __init__(self, mtcnn=False):
        """
        Initialize Haar Cascade Face Detector & Deep Learning FER Model.
        """
        # Load OpenCV default Haar Cascade Face Detector for fast real-time bounding boxes
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize FER (Facial Expression Recognition) neural network detector
        self.fer_detector = FER(mtcnn=mtcnn)
        
        # 7 Target Emotion categories
        self.emotions_list = ['surprise', 'angry', 'fear', 'happy', 'sad', 'neutral', 'disgust']
        
        # Bounding box styling constants (Neon Green: (0, 255, 102) in BGR format)
        self.GREEN_NEON = (102, 255, 0)
        self.WHITE = (255, 255, 255)
        self.DARK_BG = (15, 15, 15)

    def detect_faces_and_emotions(self, frame):
        """
        Processes a BGR image frame from webcam.
        
        Returns:
            processed_frame (np.ndarray): Frame annotated with green bounding box and emotion label tag.
            results (list): List of dicts containing face bounding box coordinates and emotion probability scores.
        """
        if frame is None:
            return frame, []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        results = []
        processed_frame = frame.copy()

        for (x, y, w, h) in faces:
            # Crop facial region of interest (ROI)
            face_roi = frame[y:y+h, x:x+w]
            
            # Predict emotion probabilities using FER model
            emotion_predictions = self.fer_detector.detect_emotions(face_roi)
            
            if emotion_predictions and len(emotion_predictions) > 0:
                raw_emotions = emotion_predictions[0]['emotions']
            else:
                # Fallback neutral distribution if ROI is unclear
                raw_emotions = {
                    'surprise': 0.05,
                    'angry': 0.05,
                    'fear': 0.05,
                    'happy': 0.10,
                    'sad': 0.05,
                    'neutral': 0.65,
                    'disgust': 0.05
                }
            
            # Find dominant emotion
            dominant_emotion = max(raw_emotions, key=raw_emotions.get)
            dominant_score = raw_emotions[dominant_emotion]

            # 1. Draw Green Frame Bounding Box (0, 255, 0)
            self._draw_bounding_box(processed_frame, x, y, w, h, dominant_emotion, dominant_score)
            
            results.append({
                'box': (x, y, w, h),
                'emotions': raw_emotions,
                'dominant_emotion': dominant_emotion,
                'dominant_score': dominant_score
            })

        return processed_frame, results

    def _draw_bounding_box(self, img, x, y, w, h, emotion_name, score):
        """
        Draws bright neon green bounding box with corner accents and emotion text label badge.
        """
        # Draw main green rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), self.GREEN_NEON, 2)
        
        # Draw Corner Accents for high-tech look
        line_len = int(min(w, h) * 0.15)
        thick = 4
        
        # Top-Left
        cv2.line(img, (x, y), (x + line_len, y), self.WHITE, thick)
        cv2.line(img, (x, y), (x, y + line_len), self.WHITE, thick)
        
        # Top-Right
        cv2.line(img, (x + w, y), (x + w - line_len, y), self.WHITE, thick)
        cv2.line(img, (x + w, y), (x + w, y + line_len), self.WHITE, thick)

        # Bottom-Left
        cv2.line(img, (x, y + h), (x + line_len, y + h), self.WHITE, thick)
        cv2.line(img, (x, y + h), (x, y + h - line_len), self.WHITE, thick)

        # Bottom-Right
        cv2.line(img, (x + w, y + h), (x + w - line_len, y + h), self.WHITE, thick)
        cv2.line(img, (x + w, y + h), (x + w, y + h - line_len), self.WHITE, thick)

        # Draw Top Text Tag Label (e.g. "HAPPY 94%")
        label_text = f"{emotion_name.upper()} {int(score * 100)}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        # Background badge pill above bounding box
        tag_y = max(y - 10, text_h + 10)
        cv2.rectangle(
            img,
            (x, tag_y - text_h - 10),
            (x + text_w + 20, tag_y + 4),
            self.DARK_BG,
            -1
        )
        cv2.rectangle(
            img,
            (x, tag_y - text_h - 10),
            (x + text_w + 20, tag_y + 4),
            self.GREEN_NEON,
            1
        )
        
        # Text label
        cv2.putText(
            img,
            label_text,
            (x + 10, tag_y - 4),
            font,
            font_scale,
            self.GREEN_NEON,
            thickness,
            cv2.LINE_AA
        )
