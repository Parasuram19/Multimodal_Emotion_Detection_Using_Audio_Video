import cv2
import dlib
import numpy as np
from deepface import DeepFace
import os
import time
import pandas as pd
import streamlit as st
import datetime
from pathlib import Path
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import io
import wave
import base64
import altair as alt
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import threading
import queue
from typing import List, NamedTuple
import asyncio

# Set page configuration
st.set_page_config(
    page_title="Emotion Analysis",
    page_icon="😀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Directory setup
DATA_DIR = "emotion_data"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "face"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "audio"), exist_ok=True)

# Define RTC configuration (use Google's STUN server)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# App state management
class SessionState:
    def __init__(self):
        self.is_recording_video = False
        self.is_recording_audio = False
        self.audio_data = []
        self.sample_rate = 16000
        self.webrtc_ctx = None
        self.audio_receiver = None
        self.frame_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.face_emotion_data = []
        self.audio_emotion_data = []
        self.face_csv_path = None
        self.audio_csv_path = None

# Initialize session state
if 'session_state' not in st.session_state:
    st.session_state.session_state = SessionState()

session_state = st.session_state.session_state

# Global model references
audio_model = None
audio_feature_extractor = None

class EmotionDetector:
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat", csv_path=None):
        """
        Initialize the emotion detector with required models
        
        Args:
            predictor_path: Path to the dlib facial landmark predictor file
            csv_path: Path to save the emotion data CSV
        """
        # Initialize dlib's face detector and facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        
        # Check if predictor file exists
        if not os.path.isfile(predictor_path):
            st.error(f"Error: {predictor_path} not found.")
            st.info("Please download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            raise FileNotFoundError(f"Could not find {predictor_path}")
            
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # Emotion labels and colors for visualization
        self.emotions = {
            'angry': (0, 0, 255),     # Red
            'disgust': (0, 140, 255), # Orange
            'fear': (0, 255, 255),    # Yellow
            'happy': (0, 255, 0),     # Green
            'sad': (255, 0, 0),       # Blue
            'surprise': (255, 0, 255),# Magenta
            'neutral': (255, 255, 255) # White
        }
        
        # For smoothing emotions over time
        self.emotion_history = []
        self.history_size = 5
        self.last_time = time.time()
        
        # CSV data handling
        self.csv_path = csv_path
        if csv_path:
            self.initialize_csv()
        
        # For tracking detection frequency
        self.last_record_time = time.time()
        self.record_interval = 1.0  # Record every 1 second
        
    def initialize_csv(self):
        """Initialize the CSV file with headers"""
        # Create necessary directories
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        
        # Check if file exists
        if not os.path.exists(self.csv_path):
            # Create a DataFrame with columns for timestamp and each emotion
            columns = ['timestamp'] + list(self.emotions.keys())
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.csv_path, index=False)
            st.success(f"Created new CSV file at {self.csv_path}")
        else:
            st.info(f"Using existing CSV file at {self.csv_path}")
            
    def save_emotion_to_csv(self, emotion_scores):
        """
        Save emotion scores to CSV with timestamp
        
        Args:
            emotion_scores: Dictionary of emotion scores
        """
        if not self.csv_path:
            return
            
        # Only record at specified intervals
        current_time = time.time()
        if current_time - self.last_record_time < self.record_interval:
            return
            
        self.last_record_time = current_time
        
        # Create a new row with current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare data for the row
        row_data = {'timestamp': timestamp}
        
        # Add emotion scores
        for emotion, score in emotion_scores.items():
            row_data[emotion] = score
            
        # Store for live display
        session_state.face_emotion_data.append(row_data)
        
        # Load existing CSV
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            
            # Append new row
            df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
        else:
            # Create new DataFrame
            df = pd.DataFrame([row_data])
        
        # Save updated DataFrame
        df.to_csv(self.csv_path, index=False)
        
    def analyze_emotion(self, frame):
        """
        Analyze emotions in the frame using DeepFace
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary with emotion scores or None on error
        """
        if frame.size == 0:
            return None
            
        try:
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            return results[0] if isinstance(results, list) else results
        except Exception as e:
            # Just return None on error without showing the error message
            return None
            
    def smooth_emotion(self, emotion_dict):
        """
        Smooth emotions over time to prevent flickering
        
        Args:
            emotion_dict: Current emotion dictionary
            
        Returns:
            Smoothed emotion dictionary
        """
        # Add current emotions to history
        self.emotion_history.append(emotion_dict)
        
        # Keep history at fixed size
        if len(self.emotion_history) > self.history_size:
            self.emotion_history.pop(0)
            
        # Average emotions over history
        smoothed = {}
        for emotion in emotion_dict:
            values = [history[emotion] for history in self.emotion_history]
            smoothed[emotion] = sum(values) / len(values)
            
        return smoothed
        
    def draw_emotion_meter(self, frame, emotions, x, y, w, h):
        """
        Draw emotion meter on the frame
        
        Args:
            frame: Input frame
            emotions: Dictionary of emotion scores
            x, y, w, h: Face bounding box
            
        Returns:
            Frame with emotion meter
        """
        # Find dominant emotion
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
        dominant_score = emotions[dominant_emotion]
        
        # Set meter dimensions
        meter_width = min(150, frame.shape[1] - x - 10)  # Ensure meter fits within frame width
        meter_height = 20
        
        # Calculate meter position
        meter_x = x
        meter_y = y - 50
        
        # Ensure meter is within frame bounds
        if meter_y < 10:  # Add some padding from top
            meter_y = y + h + 10
            
        # Ensure meter doesn't exceed bottom of frame
        if meter_y + meter_height + 60 > frame.shape[0]:  # Add space for emotion list
            meter_y = max(10, y - 80)  # Try to move it above the face
        
        # Draw background for meter
        cv2.rectangle(frame, (meter_x, meter_y), 
                    (meter_x + meter_width, meter_y + meter_height),
                    (50, 50, 50), -1)
                    
        # Draw filled portion based on score (ensure it's between 0-100%)
        filled_width = int(meter_width * min(dominant_score / 100, 1.0))
        cv2.rectangle(frame, (meter_x, meter_y),
                    (meter_x + filled_width, meter_y + meter_height),
                    self.emotions[dominant_emotion], -1)
                    
        # Draw emotion name and score
        cv2.putText(frame, f"{dominant_emotion}: {min(dominant_score, 100):.1f}%",
                  (meter_x, meter_y - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.emotions[dominant_emotion], 2)
                  
        # Draw all emotions in a list
        emotion_y = meter_y + meter_height + 20
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        
        for i, (emotion, score) in enumerate(sorted_emotions):
            if i >= 3:  # Only show top 3 emotions
                break
            # Ensure the score is displayed as a percentage between 0-100%
            score_to_display = min(score, 100)
            cv2.putText(frame, f"{emotion}: {score_to_display:.1f}%",
                      (meter_x, emotion_y + i*20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.emotions[emotion], 1)
                      
        return frame
        
    def draw_landmarks(self, frame, shape):
        """
        Draw facial landmarks on the frame
        
        Args:
            frame: Input frame
            shape: dlib facial landmarks
            
        Returns:
            Frame with landmarks
        """
        # Convert landmarks to numpy array
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        
        # Draw all landmarks
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
        # Draw outline for different facial features
        # Jawline
        for i in range(0, 16):
            pt1 = (landmarks[i][0], landmarks[i][1])
            pt2 = (landmarks[i+1][0], landmarks[i+1][1])
            cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
            
        # Eyebrows
        for i in range(17, 21):
            pt1 = (landmarks[i][0], landmarks[i][1])
            pt2 = (landmarks[i+1][0], landmarks[i+1][1])
            cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
            
        for i in range(22, 26):
            pt1 = (landmarks[i][0], landmarks[i][1])
            pt2 = (landmarks[i+1][0], landmarks[i+1][1])
            cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
            
        # Nose
        for i in range(27, 30):
            pt1 = (landmarks[i][0], landmarks[i][1])
            pt2 = (landmarks[i+1][0], landmarks[i+1][1])
            cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
            
        # Eyes
        for i in range(36, 41):
            pt1 = (landmarks[i][0], landmarks[i][1])
            pt2 = (landmarks[i+1][0], landmarks[i+1][1])
            cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
        cv2.line(frame, (landmarks[41][0], landmarks[41][1]), 
                (landmarks[36][0], landmarks[36][1]), (0, 255, 0), 1)
                
        for i in range(42, 47):
            pt1 = (landmarks[i][0], landmarks[i][1])
            pt2 = (landmarks[i+1][0], landmarks[i+1][1])
            cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
        cv2.line(frame, (landmarks[47][0], landmarks[47][1]), 
                (landmarks[42][0], landmarks[42][1]), (0, 255, 0), 1)
                
        # Mouth
        for i in range(48, 59):
            pt1 = (landmarks[i][0], landmarks[i][1])
            pt2 = (landmarks[i+1][0], landmarks[i+1][1])
            cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
        cv2.line(frame, (landmarks[59][0], landmarks[59][1]), 
                (landmarks[48][0], landmarks[48][1]), (0, 255, 0), 1)
                
        for i in range(60, 67):
            pt1 = (landmarks[i][0], landmarks[i][1])
            pt2 = (landmarks[i+1][0], landmarks[i+1][1])
            cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
        cv2.line(frame, (landmarks[67][0], landmarks[67][1]), 
                (landmarks[60][0], landmarks[60][1]), (0, 255, 0), 1)
                
        return frame
        
    def process_frame(self, frame):
        """
        Process a single frame to detect face and emotion
        
        Args:
            frame: Input video frame
            
        Returns:
            frame with emotion detection visualization and emotion data
        """
        # Create a copy of the frame
        output_frame = frame.copy()
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector(gray)
        
        # Default empty emotion data
        emotion_data = None
        
        # Process each face found
        for face in faces:
            # Get face bounding box
            x, y = face.left(), face.top()
            w, h = face.right() - x, face.bottom() - y
            
            # Extract facial landmarks
            shape = self.predictor(gray, face)
            
            # Draw landmarks if face is detected
            output_frame = self.draw_landmarks(output_frame, shape)
            
            # Get face ROI and analyze emotion
            face_roi = frame[max(0, y):min(frame.shape[0], y+h), 
                           max(0, x):min(frame.shape[1], x+w)]
            
            # Skip if face ROI is empty
            if face_roi.size == 0:
                continue
                
            # Analyze emotion
            emotion_result = self.analyze_emotion(face_roi)
            
            if emotion_result is not None and 'emotion' in emotion_result:
                # Convert emotion scores to percentages (0-100 scale)
                # DeepFace returns values between 0-1, so multiply by 100
                emotion_scores = {}
                for emotion, score in emotion_result['emotion'].items():
                    # Ensure score is between 0-100
                    emotion_scores[emotion] = min(score * 100, 100)
                
                # Smooth emotions over time
                smoothed_emotions = self.smooth_emotion(emotion_scores)
                
                # Save emotion data for CSV recording (use first detected face)
                if emotion_data is None:
                    emotion_data = smoothed_emotions
                
                # Draw rectangle around face
                cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw emotion meter
                output_frame = self.draw_emotion_meter(output_frame, smoothed_emotions, x, y, w, h)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
        
        # Add FPS counter
        cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Record emotion data to CSV if available
        if emotion_data is not None and session_state.is_recording_video:
            self.save_emotion_to_csv(emotion_data)
        
        return output_frame, emotion_data


class VideoProcessor:
    def __init__(self):
        self.detector = None
        self.is_recording = False
        self.csv_path = None
    
    def setup(self, csv_path=None):
        """Initialize the emotion detector"""
        try:
            self.detector = EmotionDetector(csv_path=csv_path)
            self.csv_path = csv_path
            return True
        except Exception as e:
            st.error(f"Error initializing detector: {str(e)}")
            return False
    
    def recv(self, frame):
        """Process video frames from WebRTC stream"""
        img = frame.to_ndarray(format="bgr24")
        
        if self.detector and self.is_recording:
            # Process frame with emotion detector
            processed_frame, emotion_data = self.detector.process_frame(img)
            
            # Add recording indicator
            height, width = processed_frame.shape[:2]
            cv2.circle(processed_frame, (width - 30, 30), 10, (0, 0, 255), -1)  # Red circle
            cv2.putText(processed_frame, "REC", (width - 70, 40),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            processed_frame = img
            # Add status text when not recording
            cv2.putText(processed_frame, "Press 'Start Recording' to begin", (20, 40),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Put frame in queue for potential saving
        if self.is_recording:
            try:
                if not session_state.frame_queue.full():
                    session_state.frame_queue.put(img)
            except:
                pass
        
        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")


class AudioProcessor:
    def __init__(self):
        self.is_recording = False
        self.sample_rate = 16000
        self.audio_buffer = []
        self.csv_path = None
    
    def recv(self, frame):
        """Process audio frames from WebRTC stream"""
        if self.is_recording:
            sound = frame.to_ndarray().copy()
            
            # Convert stereo to mono if needed
            if sound.ndim > 1:
                sound = np.mean(sound, axis=1)
            
            # Resample to target sample rate if needed
            if frame.sample_rate != self.sample_rate:
                sound = librosa.resample(sound, orig_sr=frame.sample_rate, target_sr=self.sample_rate)
            
            # Add to buffer
            self.audio_buffer.extend(sound.tolist())
            
            # Process audio chunk if buffer is large enough (e.g., every 2 seconds)
            if len(self.audio_buffer) >= self.sample_rate * 2:
                audio_chunk = np.array(self.audio_buffer[:self.sample_rate * 2])
                self.audio_buffer = self.audio_buffer[self.sample_rate:]  # Keep overlap
                
                # Add to queue for processing
                try:
                    if not session_state.audio_queue.full():
                        session_state.audio_queue.put((audio_chunk, self.sample_rate))
                except:
                    pass
                
                # Process audio for emotion
                if audio_model is not None and audio_feature_extractor is not None:
                    emotion_result = detect_audio_emotion(
                        audio_chunk, self.sample_rate, 
                        audio_model, audio_feature_extractor
                    )
                    
                    if emotion_result is not None and self.csv_path:
                        # Save to CSV and session state for display
                        self.save_audio_emotion(emotion_result)
        
        return frame
    
    def save_audio_emotion(self, emotion_result):
        """Save audio emotion data to CSV"""
        if not self.csv_path:
            return
            
        # Create a new row with current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare data for the row
        row_data = {
            'timestamp': timestamp,
            'predicted_emotion': emotion_result['predicted_emotion'],
            'confidence': emotion_result['confidence'] * 100  # Convert to percentage
        }
        
        # Add individual emotion scores
        for emotion, score in emotion_result['all_emotions'].items():
            row_data[emotion] = score * 100  # Convert to percentage
        
        # Store for live display
        session_state.audio_emotion_data.append(row_data)
        
        # Load existing CSV
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            
            # Append new row
            df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
        else:
            # Create new DataFrame with headers
            df = pd.DataFrame([row_data])
        
        # Create directory if needed
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        
        # Save updated DataFrame
        df.to_csv(self.csv_path, index=False)


def detect_audio_emotion(audio_data, sampling_rate, model, feature_extractor):
    """
    Detect emotion from audio data using the loaded model.
    
    Args:
        audio_data: NumPy array containing audio samples
        sampling_rate: Sample rate of the audio data
        model: Pre-loaded emotion classification model
        feature_extractor: Pre-loaded feature extractor for the model
        
    Returns:
        dict: Emotion predictions and probabilities
    """
    # Skip if audio data is empty
    if len(audio_data) == 0:
        return None
    
    try:
        # Ensure audio is the right length (at least 0.5 seconds)
        if len(audio_data) < sampling_rate / 2:
            return None
            
        # Create model inputs
        inputs = feature_extractor(audio_data, sampling_rate=sampling_rate, return_tensors="pt")
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Get predicted class and probabilities
        predicted_class_id = logits.argmax().item()
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        
        # Map to emotion labels
        emotion_labels = model.config.id2label if hasattr(model.config, 'id2label') else {
            0: "angry", 1: "disgust", 2: "fear", 3: "happy",
            4: "sad", 5: "surprise", 6: "neutral"
        }
        
        # Format results
        results = {
            "predicted_emotion": emotion_labels[predicted_class_id],
            "confidence": probabilities[predicted_class_id].item(),
            "all_emotions": {
                emotion_labels[i]: prob.item()
                for i, prob in enumerate(probabilities)
            }
        }
        
        return results
    except Exception as e:
        st.error(f"Error in audio emotion detection: {e}")
        return None


def load_audio_emotion_model(model_path):
    """Load audio emotion model and feature extractor"""
    global audio_model, audio_feature_extractor
    
    try:
        audio_model = AutoModelForAudioClassification.from_pretrained(model_path)
        audio_feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        st.success("Audio emotion model loaded successfully!")
        return True
    except Exception as e:
        st.error(f"Failed to load audio emotion model: {e}")
        return False


def facial_emotion_page():
    """Facial emotion detection page"""
    st.header("Facial Emotion Detection")
    
    # Sidebar settings
    st.sidebar.header("Face Detection Settings")
    
    # CSV file path
    csv_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(DATA_DIR, "face", f"face_emotion_{csv_timestamp}.csv")
    session_state.face_csv_path = csv_path
    
    # Recording interval
    record_interval = st.sidebar.slider("Recording Interval (seconds)", 0.5, 5.0, 1.0, 0.5)
    
    # Display data
    show_data = st.sidebar.checkbox("Show Live Data", True)
    data_rows = st.sidebar.slider("Number of Data Rows to Show", 5, 50, 10)
    
    # Create placeholders
    start_stop_col1, start_stop_col2 = st.columns(2)
    data_placeholder = st.empty()
    
    # Initialize video processor
    video_processor = VideoProcessor()
    
    with start_stop_col1:
        start_button = st.button("▶️ Start Recording", use_container_width=True)
    
    with start_stop_col2:
        stop_button = st.button("⏹️ Stop & Save", use_container_width=True)
    
    # Setup WebRTC component
    ctx = webrtc_streamer(
        key="face-emotion-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=lambda: video_processor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Store WebRTC context in session state
    session_state.webrtc_ctx = ctx
    
    # Setup the emotion detector
    if ctx.video_processor and not hasattr(ctx.video_processor, 'detector'):
        if ctx.video_processor.setup(csv_path=csv_path):
            st.success("Facial emotion detector initialized!")
            ctx.video_processor.detector.record_interval = record_interval
    
    # Handle start/stop recording
    if start_button and ctx.state.playing:
        session_state.is_recording_video = True
        session_state.face_emotion_data = []  # Clear previous data
        if ctx.video_processor:
            ctx.video_processor.is_recording = True
        st.success("Started recording facial emotions")
    
    if stop_button:
        session_state.is_recording_video = False
        if ctx.video_processor:
            ctx.video_processor.is_recording = False
        st.success(f"Recording stopped. Data saved to {csv_path}")
    
    # Show recording status
    if session_state.is_recording_video and ctx.state.playing:
        st.warning("🔴 Recording in progress...")
        
        # Display live data if enabled
        if show_data and session_state.face_emotion_data:
            # Convert to dataframe for display
            face_df = pd.DataFrame(session_state.face_emotion_data)
            # Show only the most recent rows
            data_placeholder.dataframe(face_df.tail(data_rows), use_container_width=True)
    elif os.path.exists(csv_path):
        # Show the saved data when not recording
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                st.write("Recorded emotion data:")
                st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Error reading saved CSV: {e}")


def audio_emotion_page():
    """Audio emotion detection page"""
    st.header("Audio Emotion Detection")
    
    # Sidebar settings
    st.sidebar.header("Audio Detection Settings")
    
    # Model path
    model_path = st.sidebar.text_input(
        "Audio Model Directory Path", 
        "wav2vec2-savee-emotion-final"
    )
    
    # Load model button
    if st.sidebar.button("Load Audio Model"):
        load_audio_emotion_model(model_path)
    
    # CSV file path
    csv_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(DATA_DIR, "audio", f"audio_emotion_{csv_timestamp}.csv")
    session_state.audio_csv_path = csv_path
    
    # Recording controls
    col1, col2 = st.columns(2)
    
    with col1:
        start_button = st.button("🎙️ Start Recording", use_container_width=True)
    
    with col2:
        stop_button = st.button("⏹️ Stop Recording", use_container_width=True)
    
    # Status placeholder
    status_placeholder = st.empty()
    data_placeholder = st.empty()
    
    # Initialize audio processor
    audio_processor = AudioProcessor()
    audio_processor.csv_path = csv_path
    
    # Setup WebRTC component for audio
    ctx = webrtc_streamer(
        key="audio-emotion-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": False, "audio": True},
        audio_processor_factory=lambda: audio_processor,
        async_processing=True,
    )
    
    # Store WebRTC context in session state
    session_state.audio_receiver = ctx
    
    # Handle start/stop recording
    # Handle start/stop recording
    if start_button and ctx.state.playing:
        session_state.is_recording_audio = True
        session_state.audio_emotion_data = []  # Clear previous data
        if ctx.audio_processor:
            ctx.audio_processor.is_recording = True
        status_placeholder.success("Started recording audio emotions")
    
    if stop_button:
        session_state.is_recording_audio = False
        if ctx.audio_processor:
            ctx.audio_processor.is_recording = False
        status_placeholder.success(f"Recording stopped. Data saved to {csv_path}")
    
    # Show recording status
    if session_state.is_recording_audio and ctx.state.playing:
        status_placeholder.warning("🔴 Recording audio...")
        
        # Display live data if enabled
        if session_state.audio_emotion_data:
            # Convert to dataframe for display
            audio_df = pd.DataFrame(session_state.audio_emotion_data)
            # Show only the most recent rows
            data_placeholder.dataframe(audio_df.tail(10), use_container_width=True)
    elif os.path.exists(csv_path):
        # Show the saved data when not recording
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                st.write("Recorded audio emotion data:")
                st.dataframe(df, use_container_width=True)
                
                # Create visualization of emotions over time
                if len(df) > 1:
                    st.subheader("Audio Emotion Over Time")
                    
                    # Melt the dataframe to get all emotions in one column
                    emotion_cols = [col for col in df.columns if col not in ['timestamp', 'predicted_emotion', 'confidence']]
                    if emotion_cols:
                        df_melt = df.melt(
                            id_vars=['timestamp'], 
                            value_vars=emotion_cols,
                            var_name='emotion', 
                            value_name='score'
                        )
                        
                        # Create line chart
                        chart = alt.Chart(df_melt).mark_line().encode(
                            x='timestamp:T',
                            y='score:Q',
                            color='emotion:N',
                            tooltip=['timestamp', 'emotion', 'score']
                        ).properties(
                            width=700,
                            height=400
                        )
                        
                        st.altair_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"Error reading saved CSV: {e}")


def visualization_page():
    """Data visualization page"""
    st.header("Emotion Data Visualization")
    
    # List available data files
    face_files = [f for f in os.listdir(os.path.join(DATA_DIR, "face")) if f.endswith('.csv')]
    audio_files = [f for f in os.listdir(os.path.join(DATA_DIR, "audio")) if f.endswith('.csv')]
    
    st.subheader("Available Data Files")
    
    col1, col2 = st.columns(2)
    
    # Face data files
    with col1:
        st.write("Face Emotion Data:")
        if face_files:
            selected_face_file = st.selectbox(
                "Select face data file",
                face_files,
                key="face_file_select"
            )
            
            if st.button("Load Face Data"):
                face_path = os.path.join(DATA_DIR, "face", selected_face_file)
                try:
                    face_df = pd.read_csv(face_path)
                    st.session_state.loaded_face_df = face_df
                    st.success(f"Loaded {len(face_df)} records")
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        else:
            st.info("No face data files found")
    
    # Audio data files
    with col2:
        st.write("Audio Emotion Data:")
        if audio_files:
            selected_audio_file = st.selectbox(
                "Select audio data file",
                audio_files,
                key="audio_file_select"
            )
            
            if st.button("Load Audio Data"):
                audio_path = os.path.join(DATA_DIR, "audio", selected_audio_file)
                try:
                    audio_df = pd.read_csv(audio_path)
                    st.session_state.loaded_audio_df = audio_df
                    st.success(f"Loaded {len(audio_df)} records")
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        else:
            st.info("No audio data files found")
    
    # Visualization section
    st.subheader("Visualizations")
    
    # Create tabs for different visualization types
    viz_tabs = st.tabs(["Face Emotions", "Audio Emotions", "Comparison"])
    
    # Face emotions tab
    with viz_tabs[0]:
        if "loaded_face_df" in st.session_state:
            face_df = st.session_state.loaded_face_df
            
            # Time series visualization
            st.write("Emotion Trends Over Time")
            
            # Get emotion columns
            emotion_cols = [col for col in face_df.columns 
                           if col != 'timestamp' and not pd.api.types.is_numeric_dtype(face_df[col])]
            
            # Melt dataframe for visualization
            face_melt = face_df.melt(
                id_vars=['timestamp'],
                value_vars=[col for col in face_df.columns if col != 'timestamp'],
                var_name='emotion',
                value_name='score'
            )
            
            # Create time series chart
            time_chart = alt.Chart(face_melt).mark_line().encode(
                x=alt.X('timestamp:T', title='Time'),
                y=alt.Y('score:Q', title='Intensity (%)'),
                color=alt.Color('emotion:N', title='Emotion'),
                tooltip=['timestamp', 'emotion', 'score']
            ).properties(
                width=700,
                height=400,
                title='Emotion Intensity Over Time'
            )
            
            st.altair_chart(time_chart, use_container_width=True)
            
            # Average emotion distribution
            st.write("Average Emotion Distribution")
            
            # Calculate averages
            emotion_avgs = face_df.drop('timestamp', axis=1).mean().reset_index()
            emotion_avgs.columns = ['emotion', 'average']
            
            # Create bar chart
            bar_chart = alt.Chart(emotion_avgs).mark_bar().encode(
                x=alt.X('emotion:N', sort='-y', title='Emotion'),
                y=alt.Y('average:Q', title='Average Intensity (%)'),
                color=alt.Color('emotion:N', legend=None),
                tooltip=['emotion', 'average']
            ).properties(
                width=700,
                height=400,
                title='Average Emotion Distribution'
            )
            
            st.altair_chart(bar_chart, use_container_width=True)
            
            # Heatmap
            st.write("Emotion Heatmap")
            
            # Pivot data for heatmap
            if len(face_df) > 1:
                # Convert timestamp to sequential numbers
                face_df['time_index'] = range(len(face_df))
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(12, 8))
                heatmap_data = face_df.drop('timestamp', axis=1)
                if 'time_index' in heatmap_data.columns:
                    heatmap_data = heatmap_data.set_index('time_index')
                
                sns.heatmap(heatmap_data, cmap="YlGnBu", ax=ax)
                plt.title("Emotion Intensity Heatmap")
                plt.xlabel("Emotions")
                plt.ylabel("Time Sequence")
                st.pyplot(fig)
        else:
            st.info("Load face data to see visualizations")
    
    # Audio emotions tab
    with viz_tabs[1]:
        if "loaded_audio_df" in st.session_state:
            audio_df = st.session_state.loaded_audio_df
            
            # Check if we have necessary columns
            if 'predicted_emotion' in audio_df.columns and 'confidence' in audio_df.columns:
                # Distribution of predicted emotions
                st.write("Distribution of Detected Emotions")
                
                emotion_counts = audio_df['predicted_emotion'].value_counts().reset_index()
                emotion_counts.columns = ['emotion', 'count']
                
                pie_chart = alt.Chart(emotion_counts).mark_arc().encode(
                    theta=alt.Theta(field='count', type='quantitative'),
                    color=alt.Color(field='emotion', type='nominal'),
                    tooltip=['emotion', 'count']
                ).properties(
                    width=400,
                    height=400,
                    title='Distribution of Detected Emotions'
                )
                
                st.altair_chart(pie_chart, use_container_width=True)
                
                # Confidence over time
                st.write("Detection Confidence Over Time")
                
                line_chart = alt.Chart(audio_df).mark_line().encode(
                    x=alt.X('timestamp:T', title='Time'),
                    y=alt.Y('confidence:Q', title='Confidence (%)'),
                    tooltip=['timestamp', 'predicted_emotion', 'confidence']
                ).properties(
                    width=700,
                    height=400,
                    title='Detection Confidence Over Time'
                )
                
                st.altair_chart(line_chart, use_container_width=True)
                
                # All emotion scores if available
                emotion_cols = [col for col in audio_df.columns 
                               if col not in ['timestamp', 'predicted_emotion', 'confidence']]
                
                if emotion_cols:
                    st.write("All Emotion Scores Over Time")
                    
                    # Melt dataframe for visualization
                    audio_melt = audio_df.melt(
                        id_vars=['timestamp'],
                        value_vars=emotion_cols,
                        var_name='emotion',
                        value_name='score'
                    )
                    
                    # Create time series chart
                    emotion_chart = alt.Chart(audio_melt).mark_line().encode(
                        x=alt.X('timestamp:T', title='Time'),
                        y=alt.Y('score:Q', title='Score (%)'),
                        color=alt.Color('emotion:N', title='Emotion'),
                        tooltip=['timestamp', 'emotion', 'score']
                    ).properties(
                        width=700,
                        height=400,
                        title='Emotion Scores Over Time'
                    )
                    
                    st.altair_chart(emotion_chart, use_container_width=True)
            else:
                st.warning("The selected audio data file doesn't contain the expected columns")
        else:
            st.info("Load audio data to see visualizations")
    
    # Comparison tab
    with viz_tabs[2]:
        if "loaded_face_df" in st.session_state and "loaded_audio_df" in st.session_state:
            st.write("Face and Audio Emotion Comparison")
            
            # Merge datasets on timestamp
            face_df = st.session_state.loaded_face_df
            audio_df = st.session_state.loaded_audio_df
            
            # Resample both datasets to align timestamps
            try:
                # Convert timestamp to datetime
                face_df['timestamp'] = pd.to_datetime(face_df['timestamp'])
                audio_df['timestamp'] = pd.to_datetime(audio_df['timestamp'])
                
                # Select common emotions if possible
                face_emotions = [col for col in face_df.columns if col != 'timestamp']
                audio_emotions = [col for col in audio_df.columns 
                                if col not in ['timestamp', 'predicted_emotion', 'confidence']]
                
                common_emotions = list(set(face_emotions) & set(audio_emotions))
                
                if common_emotions:
                    st.write("Common emotions found between face and audio data:")
                    st.write(", ".join(common_emotions))
                    
                    # Create comparison chart
                    comparison_data = []
                    
                    for emotion in common_emotions:
                        face_avg = face_df[emotion].mean()
                        audio_avg = audio_df[emotion].mean() if emotion in audio_df.columns else None
                        
                        if audio_avg is not None:
                            comparison_data.append({
                                'emotion': emotion,
                                'face': face_avg,
                                'audio': audio_avg
                            })
                    
                    if comparison_data:
                        comparison_df = pd.DataFrame(comparison_data)
                        
                        # Melt for grouped bar chart
                        comp_melt = comparison_df.melt(
                            id_vars=['emotion'],
                            value_vars=['face', 'audio'],
                            var_name='source',
                            value_name='average'
                        )
                        
                        # Create grouped bar chart
                        comp_chart = alt.Chart(comp_melt).mark_bar().encode(
                            x=alt.X('emotion:N', title='Emotion'),
                            y=alt.Y('average:Q', title='Average Intensity (%)'),
                            color=alt.Color('source:N', title='Source'),
                            tooltip=['emotion', 'source', 'average']
                        ).properties(
                            width=700,
                            height=400,
                            title='Face vs Audio Emotion Comparison'
                        )
                        
                        st.altair_chart(comp_chart, use_container_width=True)
                else:
                    st.warning("No common emotions found between face and audio data")
                    
                # Timeline comparison
                st.write("Emotion Timeline Comparison")
                st.write("This shows how emotions change over time in both face and audio data")
                
                # Create timeline visualization
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
                
                # Face emotions timeline
                face_data = face_df.set_index('timestamp')
                face_data.drop(columns=[col for col in face_data.columns if 'time_index' in col], errors='ignore', inplace=True)
                face_data.plot(ax=ax1, linewidth=2)
                ax1.set_title('Face Emotions Over Time')
                ax1.set_ylabel('Intensity (%)')
                ax1.legend(loc='upper right')
                
                # Audio emotions timeline
                audio_data = audio_df.set_index('timestamp')
                audio_cols = [col for col in audio_data.columns 
                             if col not in ['predicted_emotion', 'confidence']]
                if audio_cols:
                    audio_data[audio_cols].plot(ax=ax2, linewidth=2)
                    ax2.set_title('Audio Emotions Over Time')
                    ax2.set_ylabel('Intensity (%)')
                    ax2.set_xlabel('Time')
                    ax2.legend(loc='upper right')
                
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error comparing datasets: {e}")
        else:
            st.info("Load both face and audio data to see comparisons")


def settings_page():
    """Settings page for the application"""
    st.header("Application Settings")
    
    # Data management section
    st.subheader("Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear All Data", type="primary"):
            try:
                # Remove all CSV files
                for folder in ["face", "audio"]:
                    folder_path = os.path.join(DATA_DIR, folder)
                    for file in os.listdir(folder_path):
                        if file.endswith('.csv'):
                            os.remove(os.path.join(folder_path, file))
                st.success("All data files cleared successfully!")
            except Exception as e:
                st.error(f"Error clearing data: {e}")
    
    with col2:
        if st.button("Download All Data"):
            try:
                # Create a zip file with all data
                import zipfile
                import io
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for folder in ["face", "audio"]:
                        folder_path = os.path.join(DATA_DIR, folder)
                        for file in os.listdir(folder_path):
                            if file.endswith('.csv'):
                                file_path = os.path.join(folder_path, file)
                                zf.write(file_path, os.path.join(folder, file))
                
                # Offer download
                zip_buffer.seek(0)
                b64 = base64.b64encode(zip_buffer.read()).decode()
                href = f'<a href="data:application/zip;base64,{b64}" download="emotion_data.zip">Download ZIP File</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success("Data prepared for download!")
            except Exception as e:
                st.error(f"Error preparing download: {e}")
    
    # Model settings
    st.subheader("Model Settings")
    
    # Face model settings
    st.write("Face Model Settings")
    face_predictor_path = st.text_input(
        "Face Landmark Predictor Path",
        "shape_predictor_68_face_landmarks.dat"
    )
    
    if st.button("Check Face Model"):
        if os.path.exists(face_predictor_path):
            st.success(f"Face landmark predictor found at {face_predictor_path}")
        else:
            st.error(f"Face landmark predictor not found at {face_predictor_path}")
            st.info("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    
    # Audio model settings
    st.write("Audio Model Settings")
    audio_model_path = st.text_input(
        "Audio Emotion Model Path",
        "wav2vec2-savee-emotion-final"
    )
    
    if st.button("Check Audio Model"):
        if os.path.exists(audio_model_path):
            st.success(f"Audio emotion model found at {audio_model_path}")
            if st.button("Load Audio Model Now"):
                load_audio_emotion_model(audio_model_path)
        else:
            st.error(f"Audio emotion model not found at {audio_model_path}")
            st.info("Please provide a valid path to a Hugging Face compatible audio classification model")
    
    # App settings
    st.subheader("Application Settings")
    
    # Theme selection
    theme = st.selectbox(
        "Application Theme",
        ["Light", "Dark"],
        index=1
    )
    
    # About section
    st.subheader("About")
    st.markdown("""
    **Emotion Analysis Application**
    
    This application detects and analyzes emotions from facial expressions and audio.
    
    **Features:**
    - Real-time facial emotion detection
    - Audio emotion analysis
    - Data visualization and comparison
    - CSV data export
    
    **Required models:**
    - dlib facial landmark predictor
    - DeepFace for facial emotion detection
    - Hugging Face audio classification model
    
    **Version:** 1.0.0
    """)


# Main app
def main():
    """Main application function"""
    st.sidebar.title("Emotion Analysis")
    
    # App navigation
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["Home", "Face Emotion", "Audio Emotion", "Visualization", "Settings"]
    )
    
    if app_mode == "Home":
        st.title("Emotion Analysis Application")
        
        st.markdown("""
        Welcome to the Emotion Analysis Application! This tool helps you analyze emotions through:
        
        - **Facial expressions** - Detect emotions from your face in real-time
        - **Voice analysis** - Analyze emotions in your speech
        - **Data visualization** - Compare and visualize emotion data
        
        Use the sidebar to navigate between different modes.
        """)
        
        # Feature showcase
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Facial Emotion Detection")
            st.markdown("""
            - Detects 7 emotions: angry, disgust, fear, happy, sad, surprise, and neutral
            - Records emotions over time
            - Visualizes emotion intensity with color-coded meters
            """)
            st.button("Try Face Detection", on_click=lambda: st.session_state.update({"app_mode": "Face Emotion"}))
        
        with col2:
            st.subheader("Audio Emotion Analysis")
            st.markdown("""
            - Analyzes emotions in speech
            - Detects emotional tone and intensity
            - Works in real-time with your microphone
            """)
            st.button("Try Audio Analysis", on_click=lambda: st.session_state.update({"app_mode": "Audio Emotion"}))
        
        # Setup instructions
        st.subheader("Setup Instructions")
        st.markdown("""
        Before using the application, make sure you have:
        
        1. Installed all required dependencies
        2. Downloaded the facial landmark predictor file
        3. Set up an audio emotion classification model
        
        Visit the Settings page for more information on required models.
        """)
        
    elif app_mode == "Face Emotion":
        facial_emotion_page()
    elif app_mode == "Audio Emotion":
        audio_emotion_page()
    elif app_mode == "Visualization":
        visualization_page()
    elif app_mode == "Settings":
        settings_page()


if __name__ == "__main__":
    main()
    