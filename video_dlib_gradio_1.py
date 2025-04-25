import cv2
import dlib
import numpy as np
from deepface import DeepFace
import os
import time
import pandas as pd
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import io
import gradio as gr
import threading
import queue
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import librosa
import wave
import base64
from typing import List, NamedTuple
import json

# Directory setup
DATA_DIR = "emotion_data"
# MODEL_DIR = "wav2vec2-savee-emotion-final"  # Path to your pretrained model folder
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "face"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "audio"), exist_ok=True)

# Global variables and state management
class AppState:
    def __init__(self):
        self.is_recording_video = False
        self.is_recording_audio = False
        self.audio_data = []
        self.sample_rate = 16000
        self.frame_queue = queue.Queue(maxsize=100)
        self.audio_queue = queue.Queue(maxsize=100)
        self.face_emotion_data = []
        self.audio_emotion_data = []
        self.face_csv_path = None
        self.audio_csv_path = None
        self.recorded_audio = None  # Store complete audio for processing after recording

# Initialize app state
app_state = AppState()
MODEL_DIR = "./wav2vec2_emotion_local__/models--firdhokk--speech-emotion-recognition-with-facebook-wav2vec2-large-xlsr-53/snapshots/611e6db8ee667aa07fe66596f9fc761e036ff5b9"
# Global model references
audio_model = None
audio_feature_extractor = None
def load_audio_models():
    """Load the audio emotion recognition model from local directory"""
    global audio_model, audio_feature_extractor
    
    try:
        # Check if model directory exists
        if not os.path.exists(MODEL_DIR):
            print(f"Error: Model directory {MODEL_DIR} not found")
            return False
            
        # Check for required files
        if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
            print(f"Error: config.json not found in {MODEL_DIR}")
            return False
            
        # Load the model and feature extractor from your local folder
        print(f"Loading audio model from {MODEL_DIR}...")
        audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR)
        audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR)
        print("Audio model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading audio model: {str(e)}")
        return False

def load_audio_file(file_path):
    """Load and preprocess audio file using librosa"""
    try:
        print(f"Loading audio file: {file_path}")
        # Load audio using librosa with 16kHz sampling rate (typical for Wav2Vec2 models)
        audio, sr = librosa.load(file_path, sr=16000)
        
        # Update app state
        app_state.audio_loaded = True
        app_state.audio_data = audio
        app_state.sample_rate = sr
        
        print(f"Audio loaded successfully: {len(audio)} samples, {sr}Hz")
        return True
    except Exception as e:
        print(f"Error loading audio file: {str(e)}")
        return False

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
            print(f"Error: {predictor_path} not found.")
            print("Please download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
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
        
        # Debug info
        print("Emotion detector initialized successfully")
        
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
            print(f"Created new CSV file at {self.csv_path}")
        else:
            print(f"Using existing CSV file at {self.csv_path}")
            
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
        app_state.face_emotion_data.append(row_data)
        
        try:
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
            print(f"Saved emotion data: {row_data}")
        except Exception as e:
            print(f"Error saving to CSV: {e}")
        
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
            # Check if the frame is valid
            if frame.shape[0] < 10 or frame.shape[1] < 10:
                print("Frame too small for analysis")
                return None
                
            # Ensure frame is RGB (DeepFace requires this)
            if len(frame.shape) == 2:  # If grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:  # If RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                
            # Use DeepFace with more permissive settings
            results = DeepFace.analyze(
                frame, 
                actions=['emotion'], 
                enforce_detection=False,
                detector_backend='opencv'  # Use OpenCV for more reliable detection
            )
            
            # Debug info
            if results:
                print("DeepFace detection successful")
            else:
                print("DeepFace returned empty results")
                
            return results[0] if isinstance(results, list) else results
        except Exception as e:
            print(f"Error in emotion analysis: {str(e)}")
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
        if frame is None:
            print("Received None frame")
            return None, None
            
        # Create a copy of the frame
        output_frame = frame.copy()
        
        # Convert to grayscale for face detection
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print(f"Error converting frame: {e}")
            return output_frame, None
        
        # Detect faces
        try:
            faces = self.detector(gray)
            print(f"Detected {len(faces)} faces")
        except Exception as e:
            print(f"Error in face detection: {e}")
            faces = []
        
        # Default empty emotion data
        emotion_data = None
        
        # Process each face found
        for face in faces:
            # Get face bounding box
            x, y = face.left(), face.top()
            w, h = face.right() - x, face.bottom() - y
            
            # Draw rectangle around face
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            try:
                # Extract facial landmarks
                shape = self.predictor(gray, face)
                
                # Draw landmarks if face is detected
                output_frame = self.draw_landmarks(output_frame, shape)
            except Exception as e:
                print(f"Error in landmark prediction: {e}")
                continue
            
            # Get face ROI and analyze emotion
            try:
                # Ensure bounds are within frame
                y1 = max(0, y)
                y2 = min(frame.shape[0], y+h)
                x1 = max(0, x)
                x2 = min(frame.shape[1], x+w)
                
                face_roi = frame[y1:y2, x1:x2]
                
                # Skip if face ROI is empty
                if face_roi.size == 0:
                    print("Empty face ROI, skipping")
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
                    
                    # Draw emotion meter
                    output_frame = self.draw_emotion_meter(output_frame, smoothed_emotions, x, y, w, h)
            except Exception as e:
                print(f"Error processing face ROI: {e}")
                continue
        
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
        
        # Add FPS counter
        cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Record emotion data to CSV if available
        if emotion_data is not None and app_state.is_recording_video:
            self.save_emotion_to_csv(emotion_data)
        
        return output_frame, emotion_data

# Audio emotion analysis functions
def analyze_audio(audio_array, sample_rate=16000):
    """
    Analyze emotion in audio data
    
    Args:
        audio_array: Audio data as numpy array
        sample_rate: Sample rate of the audio
        
    Returns:
        Dictionary of emotion scores
    """
    global audio_model, audio_feature_extractor
    
    if audio_model is None or audio_feature_extractor is None:
        if not load_audio_models():
            return {"error": "Could not load audio model"}
    
    try:
        print(f"Analyzing audio of length {len(audio_array)}")
        
        # Ensure audio is the right length (1-10 seconds)
        if len(audio_array) > 10 * sample_rate:
            audio_array = audio_array[:10 * sample_rate]
        elif len(audio_array) < sample_rate:
            return {"neutral": 100.0}  # Default for too short samples
        
        # Extract features
        inputs = audio_feature_extractor(
            audio_array, 
            sampling_rate=sample_rate, 
            return_tensors="pt", 
            padding=True
        )
        
        # Get model predictions
        with torch.no_grad():
            logits = audio_model(**inputs).logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
        # Get emotion labels and scores
        id2label = audio_model.config.id2label
        emotion_scores = {}
        for i, score in enumerate(probabilities[0].tolist()):
            emotion = id2label[i]
            emotion_scores[emotion] = score * 100  # Convert to percentage
            
        print(f"Audio emotion scores: {emotion_scores}")
        return emotion_scores
        
    except Exception as e:
        print(f"Error analyzing audio: {str(e)}")
        return {"error": str(e)}

def process_complete_audio():
    """Process the complete recorded audio after stopping recording"""
    if app_state.recorded_audio is None:
        return {"error": "No recorded audio available"}
        
    sample_rate, audio_data = app_state.recorded_audio
    
    # Analyze the complete audio
    emotion_scores = analyze_audio(audio_data, sample_rate)
    
    # Save to CSV
    if app_state.audio_csv_path:
        save_audio_emotion_to_csv(emotion_scores, app_state.audio_csv_path)
        
    return emotion_scores

def save_audio_emotion_to_csv(emotion_scores, csv_path):
    """
    Save audio emotion scores to CSV
    
    Args:
        emotion_scores: Dictionary of emotion scores
        csv_path: Path to save CSV
    """
    if not csv_path:
        return
    
    # Create a new row with current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare data for the row
    row_data = {'timestamp': timestamp}
    
    # Add emotion scores
    for emotion, score in emotion_scores.items():
        row_data[emotion] = score
        
    # Store for live display
    app_state.audio_emotion_data.append(row_data)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    try:
        # Load existing CSV or create new one
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Append new row
            df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
        else:
            # Create new DataFrame with headers
            df = pd.DataFrame([row_data])
        
        # Save updated DataFrame
        df.to_csv(csv_path, index=False)
        print(f"Saved audio emotion data: {row_data}")
    except Exception as e:
        print(f"Error saving audio data to CSV: {e}")

# Function to create emotion plots
def create_emotion_plot(data, title="Emotion Analysis", max_points=30):
    """
    Create a plot of emotion data
    
    Args:
        data: List of emotion data points
        title: Plot title
        max_points: Maximum number of points to display
        
    Returns:
        Matplotlib figure
    """
    if not data:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=14)
        ax.set_title(title)
        return fig
    
    # Limit the number of data points to avoid overcrowding
    if len(data) > max_points:
        data = data[-max_points:]
    
    try:
        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame(data)
        
        # Remove any error columns
        if 'error' in df.columns:
            df = df.drop(columns=['error'])
        
        # Set timestamp as index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # Plot emotions
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Find all emotion columns
        emotion_cols = [col for col in df.columns if col not in ['timestamp', 'error']]
        
        for emotion in emotion_cols:
            if emotion in df.columns:  # Make sure column exists
                ax.plot(df.index, df[emotion], label=emotion)
        
        ax.set_title(title)
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        print(f"Created plot with {len(data)} data points")
        return fig
    except Exception as e:
        print(f"Error creating plot: {e}")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, f"Error creating plot: {str(e)}", ha='center', va='center', fontsize=12)
        ax.set_title(title)
        return fig

# Gradio interface functions
detector = None

def init_detector():
    """Initialize the emotion detector"""
    global detector
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    face_csv_path = os.path.join(DATA_DIR, "face", f"emotion_data_{timestamp}.csv")
    app_state.face_csv_path = face_csv_path
    
    try:
        detector = EmotionDetector(csv_path=face_csv_path)
        
        # Also initialize audio model
        load_audio_models()
        
        return "Detector initialized successfully!"
    except Exception as e:
        print(f"Error initializing detector: {str(e)}")
        return f"Error initializing detector: {str(e)}"

def process_video(frame):
    """
    Process video frame from Gradio webcam
    
    Args:
        frame: Input video frame
        
    Returns:
        Processed frame and status message
    """
    global detector
    
    if frame is None:
        return frame, "No video input detected"
    
    if detector is None:
        # Try to initialize detector if not done already
        init_detector()
        if detector is None:
            # Add error message to frame
            cv2.putText(frame, "Error: Detector not initialized", (20, 40),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame, "Detector not initialized"
    
    # Process frame with emotion detector
    try:
        if app_state.is_recording_video:
            processed_frame, emotion_data = detector.process_frame(frame)
            
            # Add recording indicator
            if processed_frame is not None:
                height, width = processed_frame.shape[:2]
                cv2.circle(processed_frame, (width - 30, 30), 10, (0, 0, 255), -1)  # Red circle
                cv2.putText(processed_frame, "REC", (width - 70, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                status_msg = "Recording..." if emotion_data else "Recording (no emotions detected)"
            else:
                processed_frame = frame
                status_msg = "Error processing frame"
        else:
            # Show preview with basic face detection but no recording
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detector(gray)
            
            processed_frame = frame.copy()
            for face in faces:
                # Get face bounding box
                x, y = face.left(), face.top()
                w, h = face.right() - x, face.bottom() - y
                
                # Draw rectangle around face
                cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
            # Add status text when not recording
            cv2.putText(processed_frame, "Press 'Start Recording' to begin", (20, 40),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                      
            status_msg = f"Ready - {len(faces)} faces detected"
    
        # Put frame in queue for potential saving
        if app_state.is_recording_video:
            try:
                if not app_state.frame_queue.full():
                    app_state.frame_queue.put(frame)
            except:
                pass
        
        return processed_frame, status_msg
    except Exception as e:
        print(f"Error in process_video: {e}")
        # Return original frame with error message
        if frame is not None:
            cv2.putText(frame, f"Error: {str(e)}", (20, 40),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame, f"Error: {str(e)}"

def start_recording():
    """Start recording emotion data"""
    app_state.is_recording_video = True
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    face_csv_path = os.path.join(DATA_DIR, "face", f"emotion_data_{timestamp}.csv")
    app_state.face_csv_path = face_csv_path
    
    # Reset emotion data list
    app_state.face_emotion_data = []
    
    # Initialize CSV
    if detector:
        detector.csv_path = face_csv_path
        detector.initialize_csv()
        
    return "Recording started!"

def stop_recording():
    """Stop recording emotion data"""
    app_state.is_recording_video = False
    
    # Create a summary plot
    update_face_chart()
    
    return "Recording stopped! Data saved to CSV."

def collect_audio_data(audio):
    """Collect audio data for emotion analysis
    
    Args:
        audio: Audio data tuple (sample_rate, data)
        
    Returns:
        Status message
    """
    if audio is None:
        return "No audio received"
    
    sample_rate, audio_data = audio
    
    if app_state.is_recording_audio:
        # Store audio data for later processing
        app_state.recorded_audio = (sample_rate, audio_data)
        
        # Add to queue for potential saving
        try:
            if not app_state.audio_queue.full():
                app_state.audio_queue.put(audio_data)
        except:
            pass
        
        # Analyze audio emotion
        emotion_scores = analyze_audio(audio_data, sample_rate)
        
        # Save to CSV
        if app_state.audio_csv_path:
            save_audio_emotion_to_csv(emotion_scores, app_state.audio_csv_path)
        
        return f"Recorded audio length: {len(audio_data)/sample_rate:.1f}s"
    return "Audio recording not active"

def start_audio_recording():
    """Start recording audio for emotion analysis"""
    app_state.is_recording_audio = True
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_csv_path = os.path.join(DATA_DIR, "audio", f"audio_emotion_{timestamp}.csv")
    app_state.audio_csv_path = audio_csv_path
    
    # Reset audio emotion data list
    app_state.audio_emotion_data = []
    app_state.recorded_audio = None
    
    return "Audio recording started!"

def stop_audio_recording():
    """Stop recording audio and analyze complete recording"""
    app_state.is_recording_audio = False
    
    # Process complete audio recording
    if app_state.recorded_audio:
        process_complete_audio()
        
    # Create a summary plot
    update_audio_chart()
    
    return "Audio recording stopped! Data saved to CSV."

def update_face_chart():
    """Update the face emotion chart"""
    if not app_state.face_emotion_data:
        return None
    
    fig = create_emotion_plot(app_state.face_emotion_data, "Face Emotion Analysis")
    return fig

def update_audio_chart():
    """Update the audio emotion chart"""
    if not app_state.audio_emotion_data:
        return None
    
    fig = create_emotion_plot(app_state.audio_emotion_data, "Audio Emotion Analysis")
    return fig

def export_data():
    """Export emotion data to CSV files"""
    if not app_state.face_emotion_data and not app_state.audio_emotion_data:
        return "No data to export"
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export face data
    if app_state.face_emotion_data:
        face_export_path = os.path.join(DATA_DIR, f"face_emotion_export_{timestamp}.csv")
        pd.DataFrame(app_state.face_emotion_data).to_csv(face_export_path, index=False)
    
    # Export audio data
    if app_state.audio_emotion_data:
        audio_export_path = os.path.join(DATA_DIR, f"audio_emotion_export_{timestamp}.csv")
        pd.DataFrame(app_state.audio_emotion_data).to_csv(audio_export_path, index=False)
    
    return f"Data exported to {DATA_DIR} directory"

def clear_data():
    """Clear all collected emotion data"""
    app_state.face_emotion_data = []
    app_state.audio_emotion_data = []
    app_state.recorded_audio = None
    
    # Update charts
    update_face_chart()
    update_audio_chart()
    
    return "All data cleared"

def create_combined_chart():
    """Create a combined chart of face and audio emotions"""
    if not app_state.face_emotion_data and not app_state.audio_emotion_data:
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Face emotions
    if app_state.face_emotion_data:
        df_face = pd.DataFrame(app_state.face_emotion_data)
        if 'timestamp' in df_face.columns:
            df_face['timestamp'] = pd.to_datetime(df_face['timestamp'])
            df_face.set_index('timestamp', inplace=True)
        
        emotion_cols = [col for col in df_face.columns if col not in ['timestamp', 'error']]
        for emotion in emotion_cols:
            if emotion in df_face.columns:
                ax1.plot(df_face.index, df_face[emotion], label=emotion)
        
        ax1.set_title("Face Emotion Analysis")
        ax1.set_ylim(0, 100)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No face data available", ha='center', va='center')
    
    # Audio emotions
    if app_state.audio_emotion_data:
        df_audio = pd.DataFrame(app_state.audio_emotion_data)
        if 'timestamp' in df_audio.columns:
            df_audio['timestamp'] = pd.to_datetime(df_audio['timestamp'])
            df_audio.set_index('timestamp', inplace=True)
        
        emotion_cols = [col for col in df_audio.columns if col not in ['timestamp', 'error']]
        for emotion in emotion_cols:
            if emotion in df_audio.columns:
                ax2.plot(df_audio.index, df_audio[emotion], label=emotion)
        
        ax2.set_title("Audio Emotion Analysis")
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No audio data available", ha='center', va='center')
    
    plt.tight_layout()
    return fig

def get_emotion_summary():
    """Get a summary of emotions detected"""
    summary = []
    
    # Face emotions summary
    if app_state.face_emotion_data:
        df_face = pd.DataFrame(app_state.face_emotion_data)
        face_emotions = [col for col in df_face.columns if col not in ['timestamp', 'error']]
        
        # Get average of each emotion
        face_avg = {emotion: df_face[emotion].mean() for emotion in face_emotions if emotion in df_face.columns}
        
        # Get dominant emotion
        dominant_face = max(face_avg.items(), key=lambda x: x[1])
        
        summary.append(f"Face analysis: {len(app_state.face_emotion_data)} data points")
        summary.append(f"Dominant face emotion: {dominant_face[0]} ({dominant_face[1]:.1f}%)")
        
        # Add top 3 emotions
        top_emotions = sorted(face_avg.items(), key=lambda x: x[1], reverse=True)[:3]
        summary.append("Top face emotions:")
        for emotion, score in top_emotions:
            summary.append(f"- {emotion}: {score:.1f}%")
    
    # Audio emotions summary
    if app_state.audio_emotion_data:
        df_audio = pd.DataFrame(app_state.audio_emotion_data)
        audio_emotions = [col for col in df_audio.columns if col not in ['timestamp', 'error']]
        
        # Get average of each emotion
        audio_avg = {emotion: df_audio[emotion].mean() for emotion in audio_emotions if emotion in df_audio.columns}
        
        # Get dominant emotion
        dominant_audio = max(audio_avg.items(), key=lambda x: x[1])
        
        summary.append(f"\nAudio analysis: {len(app_state.audio_emotion_data)} data points")
        summary.append(f"Dominant audio emotion: {dominant_audio[0]} ({dominant_audio[1]:.1f}%)")
        
        # Add top 3 emotions
        top_emotions = sorted(audio_avg.items(), key=lambda x: x[1], reverse=True)[:3]
        summary.append("Top audio emotions:")
        for emotion, score in top_emotions:
            summary.append(f"- {emotion}: {score:.1f}%")
    
    if not summary:
        return "No emotion data collected yet"
    
    return "\n".join(summary)

# Create the Gradio interface
def create_interface():
    """Create the Gradio interface for the emotion detector"""
    # Initialize the detector
    init_detector()
    
    with gr.Blocks(title="Emotion Detection System") as interface:
        gr.Markdown("# Multimodal Emotion Detection System")
        gr.Markdown("This system detects emotions from both face and voice in real-time.")
        
        with gr.Tabs():
            with gr.Tab("Video Emotion Detection"):
                with gr.Row():
                    with gr.Column(scale=2):
                        video_input = gr.Image(sources="webcam", streaming=True)
                        video_status = gr.Textbox(label="Status")
                    
                    with gr.Column(scale=1):
                        start_button = gr.Button("Start Recording")
                        stop_button = gr.Button("Stop Recording")
                        face_chart = gr.Plot(label="Face Emotion Analysis")
                
                video_input.stream(process_video, inputs=[video_input], 
                                   outputs=[video_input, video_status])
                start_button.click(start_recording, outputs=video_status)
                stop_button.click(stop_recording, outputs=video_status)
            
            with gr.Tab("Audio Emotion Detection"):
                with gr.Row():
                    with gr.Column(scale=2):
                        audio_input = gr.Audio(sources="microphone", streaming=True)
                        audio_status = gr.Textbox(label="Status")
                    
                    with gr.Column(scale=1):
                        start_audio_button = gr.Button("Start Audio Recording")
                        stop_audio_button = gr.Button("Stop Audio Recording")
                        audio_chart = gr.Plot(label="Audio Emotion Analysis")
                
                audio_input.stream(collect_audio_data, inputs=[audio_input], 
                                   outputs=audio_status)
                start_audio_button.click(start_audio_recording, outputs=audio_status)
                stop_audio_button.click(stop_audio_recording, outputs=audio_status)
            
            with gr.Tab("Combined Analysis"):
                with gr.Row():
                    with gr.Column(scale=2):
                        combined_chart = gr.Plot(label="Combined Emotion Analysis")
                    
                    with gr.Column(scale=1):
                        summary_text = gr.Textbox(label="Emotion Summary", lines=10)
                        update_summary_button = gr.Button("Update Summary")
                        export_button = gr.Button("Export Data")
                        clear_button = gr.Button("Clear All Data")
                
                update_summary_button.click(get_emotion_summary, outputs=summary_text)
                export_button.click(export_data, outputs=summary_text)
                clear_button.click(clear_data, outputs=summary_text)
                
                # Update combined chart
                update_combined_button = gr.Button("Update Combined Chart")
                update_combined_button.click(create_combined_chart, outputs=combined_chart)
        
        # Refresh buttons for charts
        with gr.Row():
            refresh_face_button = gr.Button("Refresh Face Chart")
            refresh_audio_button = gr.Button("Refresh Audio Chart")
        
        refresh_face_button.click(update_face_chart, outputs=face_chart)
        refresh_audio_button.click(update_audio_chart, outputs=audio_chart)
    
    return interface


def add_wellness_report_tab(interface):
    """
    Add a psychological wellness report tab to the existing emotion detection interface.
    This function analyzes exported CSV data and generates recommendations.
    
    Args:
        interface: The existing Gradio Blocks interface
        
    Returns:
        Updated interface with wellness report tab added
    """
    # Doctor recommendations database
    doctors_db = {
        "anxiety": [
            {"name": "Dr. Sarah Johnson", "specialty": "Anxiety Disorders", "experience": "15 years", "contact": "555-123-4567", "link": "https://example.com/dr-johnson"},
            {"name": "Dr. Michael Chen", "specialty": "Cognitive Behavioral Therapy", "experience": "12 years", "contact": "555-234-5678", "link": "https://example.com/dr-chen"}
        ],
        "depression": [
            {"name": "Dr. Robert Williams", "specialty": "Clinical Depression", "experience": "18 years", "contact": "555-456-7890", "link": "https://example.com/dr-williams"},
            {"name": "Dr. Jennifer Lopez", "specialty": "Mood Disorders", "experience": "14 years", "contact": "555-567-8901", "link": "https://example.com/dr-lopez"}
        ],
        "stress": [
            {"name": "Dr. Lisa Thompson", "specialty": "Stress Management", "experience": "11 years", "contact": "555-789-0123", "link": "https://example.com/dr-thompson"},
            {"name": "Dr. James Wilson", "specialty": "Work-Related Stress", "experience": "13 years", "contact": "555-890-1234", "link": "https://example.com/dr-wilson"}
        ],
        "mood_swings": [
            {"name": "Dr. Thomas Brown", "specialty": "Mood Regulation", "experience": "17 years", "contact": "555-012-3456", "link": "https://example.com/dr-brown"},
            {"name": "Dr. Amanda Garcia", "specialty": "Bipolar Disorder", "experience": "9 years", "contact": "555-901-2345", "link": "https://example.com/dr-garcia"}
        ],
        "general": [
            {"name": "Dr. Mark Johnson", "specialty": "General Psychology", "experience": "20 years", "contact": "555-345-6789", "link": "https://example.com/dr-mark-johnson"},
            {"name": "Dr. Susan Lee", "specialty": "Holistic Mental Health", "experience": "16 years", "contact": "555-456-7890", "link": "https://example.com/dr-lee"}
        ]
    }
    
    def load_exported_data():
        """
        Load the most recent exported CSV files from the DATA_DIR
        
        Returns:
            tuple: (face_df, audio_df, status_message)
        """
        try:
            # Find most recent face and audio export files
            face_files = [f for f in os.listdir(DATA_DIR) if f.startswith("face_emotion_export_")]
            audio_files = [f for f in os.listdir(DATA_DIR) if f.startswith("audio_emotion_export_")]
            
            if not face_files and not audio_files:
                return None, None, "No exported emotion data found. Please use 'Export Data' function first."
            
            face_df = None
            audio_df = None
            
            # Load face data if available
            if face_files:
                latest_face_file = max(face_files)
                face_path = os.path.join(DATA_DIR, latest_face_file)
                face_df = pd.read_csv(face_path)
                face_df['timestamp'] = pd.to_datetime(face_df['timestamp'])
            
            # Load audio data if available
            if audio_files:
                latest_audio_file = max(audio_files)
                audio_path = os.path.join(DATA_DIR, latest_audio_file)
                audio_df = pd.read_csv(audio_path)
                audio_df['timestamp'] = pd.to_datetime(audio_df['timestamp'])
            
            return face_df, audio_df, f"Loaded data from {len(face_files)} face and {len(audio_files)} audio files"
            
        except Exception as e:
            return None, None, f"Error loading data: {str(e)}"
    
    def analyze_emotional_state(face_df, audio_df):
        """
        Analyze emotional state based on face and audio data
        
        Args:
            face_df: DataFrame with face emotion data
            audio_df: DataFrame with audio emotion data
            
        Returns:
            dict: Analysis results
        """
        analysis = {
            "dominant_emotions": {},
            "emotion_stability": {},
            "overall_mood": "",
            "indicators": [],
            "wellness_score": 0,
            "recommendations": []
        }
        
        # Process face data if available
        if face_df is not None:
            # Remove non-emotion columns
            emotion_cols = [col for col in face_df.columns if col not in ['timestamp', 'error']]
            
            # Get average of each emotion
            face_emotions_avg = {emotion: face_df[emotion].mean() for emotion in emotion_cols}
            
            # Dominant emotion
            dominant_face = max(face_emotions_avg.items(), key=lambda x: x[1])
            analysis["dominant_emotions"]["face"] = {
                "emotion": dominant_face[0],
                "score": dominant_face[1]
            }
            
            # Emotion stability (standard deviation - lower means more stable)
            face_stability = {emotion: face_df[emotion].std() for emotion in emotion_cols}
            analysis["emotion_stability"]["face"] = face_stability
            
            # Check for concerning patterns
            if face_emotions_avg.get("sad", 0) > 60:
                analysis["indicators"].append("High levels of sadness detected in facial expressions")
            if face_emotions_avg.get("angry", 0) > 50:
                analysis["indicators"].append("Elevated anger detected in facial expressions")
            if face_emotions_avg.get("fear", 0) > 40:
                analysis["indicators"].append("Signs of anxiety detected in facial expressions")
            
            # Calculate stability score (lower std dev is better)
            face_stability_score = 100 - min(100, sum(face_stability.values()) * 5)
            
            # Calculate positivity score
            positive_emotions = face_emotions_avg.get("happy", 0) + face_emotions_avg.get("surprise", 0) * 0.5
            negative_emotions = face_emotions_avg.get("sad", 0) + face_emotions_avg.get("angry", 0) + face_emotions_avg.get("fear", 0) * 0.8
            
            face_positivity = 50
            if (positive_emotions + negative_emotions) > 0:
                face_positivity = (positive_emotions / (positive_emotions + negative_emotions)) * 100
                
            analysis["wellness_metrics"] = {
                "face_stability": face_stability_score,
                "face_positivity": face_positivity
            }
        
        # Process audio data if available
        if audio_df is not None:
            # Remove non-emotion columns
            emotion_cols = [col for col in audio_df.columns if col not in ['timestamp', 'error']]
            
            # Get average of each emotion
            audio_emotions_avg = {emotion: audio_df[emotion].mean() for emotion in emotion_cols}
            
            # Dominant emotion
            dominant_audio = max(audio_emotions_avg.items(), key=lambda x: x[1])
            analysis["dominant_emotions"]["audio"] = {
                "emotion": dominant_audio[0],
                "score": dominant_audio[1]
            }
            
            # Emotion stability (standard deviation - lower means more stable)
            audio_stability = {emotion: audio_df[emotion].std() for emotion in emotion_cols}
            analysis["emotion_stability"]["audio"] = audio_stability
            
            # Check for audio emotional indicators
            if audio_emotions_avg.get("sad", 0) > 60:
                analysis["indicators"].append("Sadness detected in voice tone and patterns")
            if audio_emotions_avg.get("angry", 0) > 50:
                analysis["indicators"].append("Anger detected in voice")
            if audio_emotions_avg.get("calm", 0) < 30 and audio_emotions_avg.get("calm", 0) > 0:
                analysis["indicators"].append("Low calmness levels in speech patterns")
                
            # Calculate audio stability score
            audio_stability_score = 100 - min(100, sum(audio_stability.values()) * 5)
            
            # Map audio emotions to positive/negative (adjust as needed for your model's labels)
            positive_audio = audio_emotions_avg.get("happy", 0) + audio_emotions_avg.get("calm", 0)
            negative_audio = audio_emotions_avg.get("sad", 0) + audio_emotions_avg.get("angry", 0) + audio_emotions_avg.get("fear", 0)
            
            audio_positivity = 50
            if (positive_audio + negative_audio) > 0:
                audio_positivity = (positive_audio / (positive_audio + negative_audio)) * 100
                
            # Add to analysis
            if "wellness_metrics" not in analysis:
                analysis["wellness_metrics"] = {}
                
            analysis["wellness_metrics"]["audio_stability"] = audio_stability_score
            analysis["wellness_metrics"]["audio_positivity"] = audio_positivity
        
        # Calculate overall wellness score
        metrics = analysis.get("wellness_metrics", {})
        if metrics:
            scores = list(metrics.values())
            analysis["wellness_score"] = sum(scores) / len(scores)
        
        # Determine overall mood and recommendations
        if analysis["wellness_score"] >= 75:
            analysis["overall_mood"] = "Positive"
            analysis["recommendations"].append("Maintain your current emotional well-being practices")
            analysis["recommendations"].append("Consider mindfulness exercises to further enhance emotional awareness")
            recommended_categories = ["general"]
        elif analysis["wellness_score"] >= 50:
            analysis["overall_mood"] = "Balanced"
            analysis["recommendations"].append("Regular exercise can help maintain emotional balance")
            analysis["recommendations"].append("Consider journaling to track your emotional patterns")
            recommended_categories = ["general", "stress"]
        else:
            analysis["overall_mood"] = "Needs Attention"
            
            # Add specific recommendations based on indicators
            if any("sadness" in indicator.lower() for indicator in analysis["indicators"]):
                analysis["recommendations"].append("Schedule activities that bring you joy")
                analysis["recommendations"].append("Consider talking to a professional about depressive symptoms")
                recommended_categories = ["depression", "mood_swings"]
            elif any("anger" in indicator.lower() for indicator in analysis["indicators"]):
                analysis["recommendations"].append("Practice deep breathing exercises when feeling frustrated")
                analysis["recommendations"].append("Consider anger management strategies")
                recommended_categories = ["stress", "mood_swings"]
            elif any("anxiety" in indicator.lower() or "fear" in indicator.lower() for indicator in analysis["indicators"]):
                analysis["recommendations"].append("Practice grounding techniques for anxiety")
                analysis["recommendations"].append("Reduce caffeine intake which can exacerbate anxiety")
                recommended_categories = ["anxiety", "stress"]
            else:
                analysis["recommendations"].append("Consider speaking with a mental health professional")
                analysis["recommendations"].append("Establish a regular sleep schedule to improve mood")
                recommended_categories = ["general", "mood_swings"]
        
        # Add general recommendations
        analysis["recommendations"].append("Maintain regular sleep patterns")
        analysis["recommendations"].append("Stay hydrated and maintain a balanced diet")
        
        # Add recommended doctors
        analysis["recommended_doctors"] = []
        for category in recommended_categories:
            if category in doctors_db:
                for doctor in doctors_db[category]:
                    analysis["recommended_doctors"].append(doctor)
        
        # Deduplicate doctors
        seen_doctors = set()
        unique_doctors = []
        for doctor in analysis["recommended_doctors"]:
            if doctor["name"] not in seen_doctors:
                seen_doctors.add(doctor["name"])
                unique_doctors.append(doctor)
        
        analysis["recommended_doctors"] = unique_doctors[:3]  # Limit to top 3
        
        return analysis
    
    def create_timeline_chart(face_df, audio_df):
        """
        Create a timeline chart of emotions
        
        Args:
            face_df: DataFrame with face emotion data
            audio_df: DataFrame with audio emotion data
            
        Returns:
            matplotlib figure
        """
        if face_df is None and audio_df is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=14)
            return fig
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot face emotions
        if face_df is not None:
            emotion_cols = [col for col in face_df.columns if col not in ['timestamp', 'error']]
            face_df.set_index('timestamp', inplace=True)
            for emotion in emotion_cols:
                if emotion in face_df.columns:
                    axes[0].plot(face_df.index, face_df[emotion], label=emotion)
            
            axes[0].set_title("Facial Emotions Timeline")
            axes[0].set_ylabel("Score (%)")
            axes[0].set_ylim(0, 100)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, "No facial emotion data available", 
                        ha='center', va='center', transform=axes[0].transAxes)
        
        # Plot audio emotions
        if audio_df is not None:
            emotion_cols = [col for col in audio_df.columns if col not in ['timestamp', 'error']]
            audio_df.set_index('timestamp', inplace=True)
            for emotion in emotion_cols:
                if emotion in audio_df.columns:
                    axes[1].plot(audio_df.index, audio_df[emotion], label=emotion)
            
            axes[1].set_title("Voice Emotions Timeline")
            axes[1].set_ylabel("Score (%)")
            axes[1].set_ylim(0, 100)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, "No audio emotion data available", 
                        ha='center', va='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        return fig
    
    def create_wellness_gauge(wellness_score):
        """
        Create a gauge chart for wellness score
        
        Args:
            wellness_score: Overall wellness score
            
        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
        
        # Gauge settings
        theta = np.linspace(0.25 * np.pi, -0.75 * np.pi, 100)
        r = np.ones_like(theta)
        
        # Background colors for different score ranges
        cmap = plt.cm.RdYlGn
        colors = cmap(np.linspace(0, 1, 100))
        
        # Create colorful background
        bars = ax.bar(theta, r, width=np.pi*1.5/100, color=colors, alpha=0.8)
        
        # Score needle
        score_angle = 0.25 * np.pi - (wellness_score / 100) * np.pi * 1.5
        ax.plot([0, score_angle], [0, 0.9], 'k-', lw=3)
        ax.plot([score_angle], [0.9], 'ko', ms=10)
        
        # Add wellness score text
        ax.text(0, 0, f"{wellness_score:.1f}", ha='center', va='center', 
                fontsize=24, fontweight='bold')
        
        # Remove axes components
        ax.set_yticks([])
        ax.set_xticks([0.25*np.pi, 0.125*np.pi, 0, -0.125*np.pi, -0.25*np.pi, -0.375*np.pi, -0.5*np.pi, -0.625*np.pi, -0.75*np.pi])
        ax.set_xticklabels(['0', '10', '20', '30', '40', '50', '60', '70', '80'])
        ax.grid(False)
        
        ax.set_title("Psychological Wellness Score", pad=20, fontsize=16)
        
        return fig
    
    def generate_wellness_report():
        """
        Generate psychological wellness report
        
        Returns:
            tuple: (markdown_report, timeline_chart, wellness_gauge, status)
        """
        # Load the exported data
        face_df, audio_df, status = load_exported_data()
        
        if face_df is None and audio_df is None:
            return status, None, None, status
        
        # Analyze emotional state
        analysis = analyze_emotional_state(face_df, audio_df)
        
        # Create charts
        timeline_chart = create_timeline_chart(face_df, audio_df)
        wellness_gauge = create_wellness_gauge(analysis["wellness_score"])
        
        # Generate markdown report
        report = f"""# Psychological Wellness Report
Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}


## Overall Assessment
**Wellness Score:** {analysis["wellness_score"]:.1f}/100  
**Emotional State:** {analysis["overall_mood"]}

## Emotional Analysis

"""
        
        # Add dominant emotions
        if "face" in analysis["dominant_emotions"]:
            report += f"**Dominant Facial Emotion:** {analysis['dominant_emotions']['face']['emotion']} ({analysis['dominant_emotions']['face']['score']:.1f}%)\n\n"
        
        if "audio" in analysis["dominant_emotions"]:
            report += f"**Dominant Voice Emotion:** {analysis['dominant_emotions']['audio']['emotion']} ({analysis['dominant_emotions']['audio']['score']:.1f}%)\n\n"
        
        # Add indicators
        if analysis["indicators"]:
            report += "## Observations\n\n"
            for indicator in analysis["indicators"]:
                report += f"- {indicator}\n"
            report += "\n"
        
        # Add recommendations
        report += "## Recommendations\n\n"
        for recommendation in analysis["recommendations"]:
            report += f"- {recommendation}\n"
        report += "\n"
        
        # Add recommended specialists
        if analysis["recommended_doctors"]:
            report += "## Recommended Specialists\n\n"
            for doctor in analysis["recommended_doctors"]:
                report += f"### {doctor['name']}\n"
                report += f"**Specialty:** {doctor['specialty']}\n"
                report += f"**Experience:** {doctor['experience']}\n"
                report += f"**Contact:** {doctor['contact']}\n"
                report += f"**Profile:** [{doctor['name']}]({doctor['link']})\n\n"
        
        report += """
## Disclaimer
This report is generated based on automated emotion analysis and should not be considered a clinical diagnosis. 
For accurate assessment and treatment, please consult with a licensed healthcare professional.
"""
        
        return report, timeline_chart, wellness_gauge, "Wellness report generated successfully!"
    
    # Add the wellness report tab to the interface
    with interface:
        with gr.Tab("Psychological Wellness Report"):
            with gr.Row():
                generate_report_button = gr.Button("Generate Wellness Report")
            
            with gr.Row():
                with gr.Column(scale=2):
                    report_markdown = gr.Markdown("Please click 'Generate Wellness Report' to analyze your emotional data.")
            
            with gr.Row():
                with gr.Column():
                    timeline_plot = gr.Plot(label="Emotional Timeline")
                with gr.Column():
                    wellness_gauge_plot = gr.Plot(label="Wellness Score")
            
            with gr.Row():
                report_status = gr.Textbox(label="Status")
            
            # Connect the button to the generate function
            generate_report_button.click(
                generate_wellness_report,
                outputs=[report_markdown, timeline_plot, wellness_gauge_plot, report_status]
            )
    
    return interface



#################################################################################################################################################

def add_feedback_tab(interface):
    """
    Add a psychological feedback questionnaire tab to the existing emotion detection interface.
    This function generates questions based on detected emotions and allows patients to respond.
    
    Args:
        interface: The existing Gradio Blocks interface
        
    Returns:
        Updated interface with feedback questionnaire tab added
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os
    import datetime
    import io
    import base64
    from matplotlib.colors import LinearSegmentedColormap
    
    # Question banks organized by emotional categories
    question_banks = {
        "anxiety": [
            "How often do you feel restless or on edge?",
            "Do you have difficulty controlling your worries?",
            "How frequently do you experience physical symptoms of anxiety (racing heart, sweating, etc.)?",
            "To what extent do your worries interfere with daily activities?",
            "How often do you avoid situations due to anxiety?",
            "Do you experience difficulty concentrating due to worrying thoughts?",
            "How often do you feel irritable when anxious?",
            "Do you experience sleep disturbances related to anxiety?",
            "How quickly do you become overwhelmed in stressful situations?",
            "To what extent do you catastrophize potential outcomes?"
        ],
        "depression": [
            "How often do you feel down or hopeless?",
            "Have you lost interest or pleasure in activities you used to enjoy?",
            "How often do you experience changes in appetite or weight?",
            "Do you have difficulty sleeping or sleep excessively?",
            "How frequently do you feel tired or lacking energy?",
            "Do you experience feelings of worthlessness or excessive guilt?",
            "How often do you have trouble concentrating?",
            "Have you noticed changes in your movement speed (slower or agitated)?",
            "How often do you think about death or have suicidal thoughts?",
            "To what extent do your feelings affect your ability to function day-to-day?"
        ],
        "anger": [
            "How often do you feel you cannot control your anger?",
            "Do small issues trigger significant anger responses?",
            "How frequently do you express anger verbally or physically?",
            "Do you find yourself holding grudges after conflicts?",
            "How often do you feel regret after expressing anger?",
            "Do you notice physical symptoms when angry (racing heart, tension)?",
            "How frequently do others comment on your anger issues?",
            "To what extent does anger interfere with your relationships?",
            "How quickly do you become frustrated in challenging situations?",
            "Do you have effective strategies for managing anger?"
        ],
        "stress": [
            "How often do you feel overwhelmed by your responsibilities?",
            "Do you experience physical symptoms of stress (headaches, muscle tension)?",
            "How frequently do you feel unable to cope with demands?",
            "Do you have difficulty relaxing or unwinding?",
            "How often do you feel irritable or impatient under stress?",
            "Do you notice changes in your eating habits when stressed?",
            "How frequently do you use substances to manage stress?",
            "To what extent does stress affect your sleep quality?",
            "How often do you feel mentally exhausted?",
            "Do you practice stress management techniques regularly?"
        ],
        "happiness": [
            "How often do you feel satisfied with your life?",
            "Do you experience feelings of joy in your daily activities?",
            "How frequently do you engage in activities purely for enjoyment?",
            "Do you maintain a positive outlook even during challenges?",
            "How often do you express gratitude for aspects of your life?",
            "Do you have meaningful connections with others?",
            "How frequently do you laugh or experience humor?",
            "To what extent do you feel your life has purpose?",
            "How often do you feel energized and motivated?",
            "Do you take time for self-care and personal growth?"
        ],
        "general": [
            "How would you rate your overall emotional well-being?",
            "Do you feel able to manage your emotions effectively?",
            "How often do you practice mindfulness or emotional awareness?",
            "Do you have a support system you can rely on?",
            "How frequently do you engage in self-reflection about your feelings?",
            "To what extent do you feel your emotions are balanced?",
            "How often do your emotions interfere with daily functioning?",
            "Do you notice patterns or triggers for your emotional responses?",
            "How would you rate your emotional resilience?",
            "Do you feel equipped with strategies to regulate your emotions?"
        ]
    }
    
    # Answer options for multiple choice questions
    answer_options = [
        ["Never", "Rarely", "Sometimes", "Often", "Always"],
        ["Not at all", "Slightly", "Moderately", "Considerably", "Extremely"],
        ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
        ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"],
        ["Never", "Monthly", "Weekly", "Several times weekly", "Daily"]
    ]
    
    # Recommendations based on total scores
    recommendation_ranges = {
        (0, 10): [
            "Your responses suggest minimal emotional challenges.",
            "Continue maintaining healthy emotional practices.",
            "Regular exercise and social connections can further enhance well-being.",
            "Consider mindfulness practices to build on your emotional strength."
        ],
        (11, 20): [
            "Your responses indicate mild emotional challenges.",
            "Regular breaks throughout the day may help maintain balance.",
            "Consider incorporating relaxation techniques into your routine.",
            "Maintaining a gratitude journal could enhance positive emotions."
        ],
        (21, 30): [
            "Your responses suggest moderate emotional challenges.",
            "Consider speaking with a trusted friend about your feelings.",
            "Mindfulness meditation may help manage emotional responses.",
            "Regular physical activity can improve mood and reduce stress.",
            "Establishing consistent sleep patterns may help stabilize emotions."
        ],
        (31, 40): [
            "Your responses indicate significant emotional challenges.",
            "Consider consulting with a mental health professional.",
            "Developing a self-care routine is highly recommended.",
            "Learning specific coping strategies for your emotions could be beneficial.",
            "Joining a support group might provide valuable community support."
        ],
        (41, 50): [
            "Your responses suggest severe emotional challenges.",
            "Professional mental health support is strongly recommended.",
            "A comprehensive assessment might help identify specific treatment needs.",
            "Developing crisis management strategies is important.",
            "Consider discussing medication options with a healthcare provider if appropriate."
        ]
    }
    
    # Define category colors for visualization
    category_colors = {
        "anxiety": "#FF9999",    # Light red
        "depression": "#9999FF", # Light blue
        "anger": "#FF6666",      # Darker red
        "stress": "#FFCC99",     # Light orange
        "happiness": "#99FF99",  # Light green
        "general": "#CCCCCC"     # Light gray
    }
    
    def load_emotional_data():
        """
        Load the most recent exported CSV files from the DATA_DIR
        
        Returns:
            tuple: (face_df, audio_df, status_message)
        """
        try:
            # Find most recent face and audio export files
            face_files = [f for f in os.listdir(DATA_DIR) if f.startswith("face_emotion_export_")]
            audio_files = [f for f in os.listdir(DATA_DIR) if f.startswith("audio_emotion_export_")]
            
            if not face_files and not audio_files:
                return None, None, "No exported emotion data found. Please use 'Export Data' function first."
            
            face_df = None
            audio_df = None
            
            # Load face data if available
            if face_files:
                latest_face_file = max(face_files)
                face_path = os.path.join(DATA_DIR, latest_face_file)
                face_df = pd.read_csv(face_path)
                face_df['timestamp'] = pd.to_datetime(face_df['timestamp'])
            
            # Load audio data if available
            if audio_files:
                latest_audio_file = max(audio_files)
                audio_path = os.path.join(DATA_DIR, latest_audio_file)
                audio_df = pd.read_csv(audio_path)
                audio_df['timestamp'] = pd.to_datetime(audio_df['timestamp'])
            
            return face_df, audio_df, f"Loaded data from {len(face_files)} face and {len(audio_files)} audio files"
            
        except Exception as e:
            return None, None, f"Error loading data: {str(e)}"
    
    def determine_emotional_profile(face_df, audio_df):
        """
        Determine the emotional profile based on face and audio data
        
        Args:
            face_df: DataFrame with face emotion data
            audio_df: DataFrame with audio emotion data
            
        Returns:
            dict: Emotional profile with dominant categories
        """
        profile = {
            "anxiety": 0,
            "depression": 0,
            "anger": 0,
            "stress": 0,
            "happiness": 0
        }
        
        # Process face data if available
        if face_df is not None:
            # Simple mapping from detected emotions to our categories
            emotion_mapping = {
                "happy": "happiness",
                "sad": "depression",
                "angry": "anger", 
                "fear": "anxiety",
                "surprise": "anxiety",
                "disgust": "anger",
                "neutral": None  # Don't map neutral to any specific category
            }
            
            # Calculate average for each emotion
            for emotion, category in emotion_mapping.items():
                if emotion in face_df.columns and category is not None:
                    avg_value = face_df[emotion].mean()
                    profile[category] += avg_value * 0.5  # Face contributes 50% to profile
        
        # Process audio data if available
        if audio_df is not None:
            # Simple mapping from detected emotions to our categories
            emotion_mapping = {
                "happy": "happiness",
                "sad": "depression",
                "angry": "anger",
                "fear": "anxiety",
                "calm": "happiness"  # Map calm to happiness/positive category
            }
            
            # Calculate average for each emotion
            for emotion, category in emotion_mapping.items():
                if emotion in audio_df.columns and category is not None:
                    avg_value = audio_df[emotion].mean()
                    profile[category] += avg_value * 0.5  # Audio contributes 50% to profile
        
        # Always ensure general category is included
        profile["general"] = 100  # Always include general questions
        
        return profile
    
    def generate_questions(emotional_profile):
        """
        Generate questions based on emotional profile
        
        Args:
            emotional_profile: Dict with scores for each emotional category
            
        Returns:
            list: 10 selected questions
        """
        selected_questions = []
        
        if sum(emotional_profile.values()) == 0:
            # No emotion data, use general questions
            selected_questions = question_banks["general"][:10]
        else:
            # Calculate how many questions to take from each category
            # Sort categories by score (highest first)
            sorted_categories = sorted(
                emotional_profile.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Always include the top 3 categories
            top_categories = [cat for cat, _ in sorted_categories[:3] if cat in question_banks]
            
            # Always include general category
            if "general" not in top_categories:
                top_categories.append("general")
            
            # Calculate questions per category
            total_questions = 10
            questions_per_category = total_questions // len(top_categories)
            remainder = total_questions % len(top_categories)
            
            # Distribute questions
            for i, category in enumerate(top_categories):
                # Add an extra question to early categories if we have remainder
                extra = 1 if i < remainder else 0
                
                # Get questions from this category
                category_questions = question_banks.get(category, question_banks["general"])
                
                # Add questions from this category
                num_questions = questions_per_category + extra
                selected_questions.extend(category_questions[:num_questions])
            
            # If we have less than 10 questions, add more from general
            while len(selected_questions) < 10:
                for q in question_banks["general"]:
                    if q not in selected_questions:
                        selected_questions.append(q)
                        break
            
            # Trim to exactly 10 questions
            selected_questions = selected_questions[:10]
        
        return selected_questions
    
    def map_question_to_category(question):
        """
        Map a question back to its emotional category
        
        Args:
            question: Question text to map
            
        Returns:
            str: Category name
        """
        for category, questions in question_banks.items():
            if question in questions:
                return category
        return "general"  # Default if not found
    
    def save_feedback_responses(responses, questions):
        """
        Save feedback responses to a file
        
        Args:
            responses: List of response values (0-4)
            questions: List of questions that were asked
            
        Returns:
            str: Status message
        """
        try:
            # Create a timestamp for the filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(DATA_DIR, f"feedback_responses_{timestamp}.csv")
            
            # Create dataframe with responses
            df = pd.DataFrame({
                "question": questions,
                "response_value": responses,
                "response_text": [answer_options[0][r] for r in responses],
                "category": [map_question_to_category(q) for q in questions],
                "timestamp": datetime.datetime.now()
            })
            
            # Save to CSV
            df.to_csv(filename, index=False)
            
            return f"Responses saved to {filename}", filename
        except Exception as e:
            return f"Error saving responses: {str(e)}", None
    
    def generate_feedback_recommendations(responses):
        """
        Generate recommendations based on responses
        
        Args:
            responses: List of response values (0-4)
            
        Returns:
            str: Recommendations
        """
        # Calculate total score (0-4 for each question)
        total_score = sum(responses)
        
        # Find the appropriate recommendation range
        recommendations = []
        for score_range, rec_list in recommendation_ranges.items():
            if score_range[0] <= total_score <= score_range[1]:
                recommendations = rec_list
                break
        
        # If no match (shouldn't happen), use middle range
        if not recommendations:
            recommendations = recommendation_ranges.get((21, 30), ["No specific recommendations available."])
        
        # Format as HTML
        recommendation_text = f"""
        <div style="padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
            <h3>Your Score: {total_score}/40</h3>
            <h4>Recommendations:</h4>
            <ul>
        """
        
        for rec in recommendations:
            recommendation_text += f"<li>{rec}</li>"
        
        recommendation_text += """
            </ul>
            <p><em>Note: These recommendations are based on your responses and are not a clinical diagnosis. 
            Please consult with a healthcare professional for personalized advice.</em></p>
        </div>
        """
        
        return recommendation_text
    
    def create_response_bar_chart(responses, questions):
        """
        Create a bar chart of responses - dark mode compatible
        
        Args:
            responses: List of response values (0-4)
            questions: List of questions that were asked
            
        Returns:
            str: HTML with embedded image
        """
        # Use dark background style
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 8))
        
        # Map questions to categories for coloring
        categories = [map_question_to_category(q) for q in questions]
        
        # Create shortened questions for better display
        short_questions = []
        for q in questions:
            if len(q) > 40:
                short_questions.append(q[:37] + "...")
            else:
                short_questions.append(q)
        
        # Create color map - use brighter colors for dark mode
        category_colors_dark = {
            "anxiety": "#FF5050",    # Brighter red
            "depression": "#5050FF", # Brighter blue
            "anger": "#FF3333",      # Brighter red
            "stress": "#FFAA33",     # Brighter orange
            "happiness": "#33FF33",  # Brighter green
            "general": "#AAAAAA"     # Light gray
        }
        
        # Create color map
        colors = [category_colors_dark.get(cat, "#AAAAAA") for cat in categories]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(short_questions))
        plt.barh(y_pos, responses, align='center', color=colors)
        plt.yticks(y_pos, short_questions, color='white')
        plt.xticks(range(5), ["Never", "Rarely", "Sometimes", "Often", "Always"], color='white')
        plt.xlabel('Response', color='white')
        plt.title('Feedback Responses by Question', color='white')
        
        # Add a legend
        legend_items = []
        for category, color in category_colors_dark.items():
            if category in categories:
                legend_items.append(plt.Rectangle((0,0), 1, 1, fc=color, edgecolor='none'))
        
        legend_categories = [cat.capitalize() for cat in category_colors_dark.keys() if cat in categories]
        if legend_items:
            plt.legend(legend_items, legend_categories, loc='best')
        
        # Add grid lines for readability
        plt.grid(axis='x', linestyle='--', alpha=0.3, color='gray')
        
        # Tight layout to ensure everything fits
        plt.tight_layout()
        
        # Convert plot to base64 for HTML embedding
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, facecolor='#121212')
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return f'<img src="data:image/png;base64,{img_str}" style="max-width:100%">'

    def create_category_radar_chart(responses, questions):
        """
        Create a radar chart showing responses by emotional category - dark mode compatible
        
        Args:
            responses: List of response values (0-4)
            questions: List of questions that were asked
            
        Returns:
            str: HTML with embedded image
        """
        # Map questions to categories
        categories = [map_question_to_category(q) for q in questions]
        
        # Calculate average score per category
        category_scores = {}
        for cat in set(categories):
            cat_responses = [responses[i] for i, c in enumerate(categories) if c == cat]
            if cat_responses:
                category_scores[cat] = sum(cat_responses) / len(cat_responses)
        
        # If we don't have enough categories, return empty
        if len(category_scores) <= 1:
            return "<p>Not enough different question categories for radar chart.</p>"
        
        # Use dark background style
        plt.style.use('dark_background')
        
        # Create radar chart
        fig = plt.figure(figsize=(8, 8), facecolor='#121212')
        ax = fig.add_subplot(111, polar=True)
        
        # Set axis colors for dark mode
        ax.set_facecolor('#121212')
        ax.spines['polar'].set_color('white')
        ax.tick_params(axis='both', colors='white')
        
        # Arrange categories in a circle
        cats = list(category_scores.keys())
        N = len(cats)
        
        # Compute radar angles
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Add values to the chart
        values = list(category_scores.values())
        values += values[:1]  # Close the loop
        
        # Define brighter colors for dark mode
        category_colors_dark = {
            "anxiety": "#FF5050",    # Brighter red
            "depression": "#5050FF", # Brighter blue
            "anger": "#FF3333",      # Brighter red
            "stress": "#FFAA33",     # Brighter orange
            "happiness": "#33FF33",  # Brighter green
            "general": "#DDDDDD"     # Light gray
        }
        
        # Draw the chart with a bright color
        line_color = category_colors_dark.get(list(category_scores.keys())[0], "#33AAFF")
        ax.plot(angles, values, linewidth=2, linestyle='solid', c=line_color)
        
        # Fill the area
        ax.fill(angles, values, alpha=0.25, color=line_color)
        
        # Set category labels
        plt.xticks(angles[:-1], [c.capitalize() for c in cats], color='white')
        
        # Set y-axis limits
        ax.set_ylim(0, 4)
        
        # Add circles at each increment
        for i in range(1, 5):
            circle = plt.Circle((0, 0), i, transform=ax.transData._b, fill=False, edgecolor='gray', alpha=0.3)
            ax.add_patch(circle)
        
        # Add title with white text
        plt.title('Emotional Category Profile', size=15, y=1.1, color='white')
        
        # Convert plot to base64 for HTML embedding
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, facecolor='#121212')
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return f'<img src="data:image/png;base64,{img_str}" style="max-width:100%">'
    
    def create_detailed_report(responses, questions):
        """
        Create a detailed report with analysis of responses
        
        Args:
            responses: List of response values (0-4)
            questions: List of questions that were asked
            
        Returns:
            str: HTML report
        """
        # Map questions to categories
        categories = [map_question_to_category(q) for q in questions]
        
        # Calculate statistics
        total_score = sum(responses)
        avg_score = total_score / len(responses) if responses else 0
        
        # Calculate category scores
        category_scores = {}
        for cat in set(categories):
            cat_responses = [responses[i] for i, c in enumerate(categories) if c == cat]
            if cat_responses:
                category_scores[cat] = sum(cat_responses) / len(cat_responses)
        
        # Sort categories by score (descending)
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Determine highest scoring category
        highest_category = sorted_categories[0][0] if sorted_categories else "general"
        
        # Determine level of concern
        concern_levels = ["Minimal", "Mild", "Moderate", "Significant", "Severe"]
        concern_index = min(4, total_score // 10)
        concern_level = concern_levels[concern_index]
        
        # Create the HTML report
        report = f"""
        <div style="padding: 20px; background-color: #f8f9fa; border-radius: 10px; font-family: Arial, sans-serif;">
            <h2>Emotional Feedback Analysis Report</h2>
            
            <div style="margin: 20px 0; padding: 15px; background-color: #ffffff; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <h3>Summary</h3>
                <p><strong>Total Score:</strong> {total_score}/40</p>
                <p><strong>Average Response:</strong> {avg_score:.1f}/4</p>
                <p><strong>Concern Level:</strong> <span style="color: {'#ff0000' if concern_index >= 3 else '#ff9900' if concern_index >= 2 else '#009900'}">{concern_level}</span></p>
                <p><strong>Primary Emotional Category:</strong> {highest_category.capitalize()}</p>
            </div>
            
            <div style="margin: 20px 0; padding: 15px; background-color: #ffffff; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <h3>Emotional Category Scores</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="background-color: #f0f0f0;">
                        <th style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd;">Category</th>
                        <th style="padding: 8px; text-align: center; border-bottom: 1px solid #ddd;">Average Score</th>
                        <th style="padding: 8px; text-align: center; border-bottom: 1px solid #ddd;">Visual</th>
                    </tr>
        """
        
        # Add rows for each category
        for cat, score in sorted_categories:
            # Create a simple visual bar
            bar_width = int(score * 25)  # 0-4 scale to 0-100%
            bar_color = category_colors.get(cat, "#CCCCCC")
            
            report += f"""
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{cat.capitalize()}</td>
                    <td style="padding: 8px; text-align: center; border-bottom: 1px solid #ddd;">{score:.1f}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">
                        <div style="background-color: #f0f0f0; width: 100%; height: 20px; border-radius: 3px;">
                            <div style="background-color: {bar_color}; width: {bar_width}%; height: 20px; border-radius: 3px;"></div>
                        </div>
                    </td>
                </tr>
            """
        
        report += """
                </table>
            </div>
            
            <div style="margin: 20px 0; padding: 15px; background-color: #ffffff; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <h3>Detailed Responses</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="background-color: #f0f0f0;">
                        <th style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd;">Question</th>
                        <th style="padding: 8px; text-align: center; border-bottom: 1px solid #ddd;">Response</th>
                        <th style="padding: 8px; text-align: center; border-bottom: 1px solid #ddd;">Category</th>
                    </tr>
        """
        
        # Add rows for each question
        for i, (question, response, category) in enumerate(zip(questions, responses, categories)):
            response_text = answer_options[0][response]
            
            # Alternate row colors for readability
            bg_color = "#f9f9f9" if i % 2 == 0 else "#ffffff"
            
            report += f"""
                <tr style="background-color: {bg_color};">
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{question}</td>
                    <td style="padding: 8px; text-align: center; border-bottom: 1px solid #ddd;">{response_text}</td>
                    <td style="padding: 8px; text-align: center; border-bottom: 1px solid #ddd; color: {category_colors.get(category, '#000000')}">{category.capitalize()}</td>
                </tr>
            """
        
        report += """
                </table>
            </div>
            
            <div style="margin: 20px 0; font-size: 12px; color: #666;">
                <p><em>Note: This report is generated based on your responses and is not a clinical diagnosis. 
                Please consult with a healthcare professional for personalized advice.</em></p>
                <p><em>Generated on: """ + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</em></p>
            </div>
        </div>
        """
        
        return report
    
    def run_feedback_questionnaire():
        """
        Main function to run the feedback questionnaire
        
        Returns:
            tuple: Contains all required outputs in the correct order
        """
        # Load emotional data
        face_df, audio_df, status = load_emotional_data()
        
        # This is the standard set of answer options we'll use for all questions
        standard_answers = answer_options[0]  # ["Never", "Rarely", "Sometimes", "Often", "Always"]
        
        if face_df is None and audio_df is None:
            # Return default questions with the standard answer options
            default_questions = ["Please export emotion data first"] * 10
            
            return (
                default_questions,  # state
                standard_answers, standard_answers, standard_answers, standard_answers, standard_answers,
                standard_answers, standard_answers, standard_answers, standard_answers, standard_answers,
                status
            )
        
        # Determine emotional profile
        emotional_profile = determine_emotional_profile(face_df, audio_df)
        
        # Generate questions
        questions = generate_questions(emotional_profile)
        
        # Return questions with standard answer options
        return (
            questions,  # state
            standard_answers, standard_answers, standard_answers, standard_answers, standard_answers,
            standard_answers, standard_answers, standard_answers, standard_answers, standard_answers,
            "Questionnaire generated based on detected emotions"
        )
    
    def update_questions(questions):
        """
        Update question textboxes
        
        Args:
            questions: List of questions
            
        Returns:
            tuple: All questions in the correct order
        """
        # Return state and all question textboxes
        return (
            questions,  # state
            questions[0], questions[1], questions[2], questions[3], questions[4],
            questions[5], questions[6], questions[7], questions[8], questions[9]
        )
    
    def process_feedback_submission(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, questions):
        """
        Process the submitted feedback
        
        Args:
            q1-q10: Response values (selected option text)
            questions: List of questions that were asked
            
        Returns:
            tuple: (recommendations_html, status, bar_chart_html, radar_chart_html, detailed_report_html)
        """
        # Combine all responses - convert selection to integer index
        responses = []
        standard_options = answer_options[0]
        
        for resp in [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]:
            try:
                # Try to find the index of the response in the standard options
                if resp in standard_options:
                    responses.append(standard_options.index(resp))
                else:
                    # Default to 0 if not found
                    responses.append(0)
            except:
                # Default to 0 if there's an error
                responses.append(0)
        
        # Save responses
        save_status, filename = save_feedback_responses(responses, questions)
        
        # Generate recommendations
        recommendations = generate_feedback_recommendations(responses)
        
        # Generate visualization charts
        bar_chart = create_response_bar_chart(responses, questions)
        radar_chart = create_category_radar_chart(responses, questions)
        
        # Generate detailed report
        detailed_report = create_detailed_report(responses, questions)
        
        return recommendations, save_status, bar_chart, radar_chart, detailed_report
    
    # Add the feedback tab to the interface
    with interface:
        with gr.Tab("Emotional Feedback Questionnaire"):
            with gr.Row():
                load_questionnaire_button = gr.Button("Generate Questionnaire")
            
            with gr.Row():
                feedback_status = gr.Textbox(label="Status", value="Click 'Generate Questionnaire' to begin")
            
            # Create question components
            questions_state = gr.State([])
            
            # Create rows for questions and answers
            question_components = []
            answer_components = []
            
            # Standard answer options
            standard_options = answer_options[0]  # ["Never", "Rarely", "Sometimes", "Often", "Always"]
            
            for i in range(10):
                
                with gr.Row():
                    q_label = gr.Textbox(label=f"Question {i+1}", value="", interactive=False)
                    q_response = gr.Dropdown(choices=standard_options, value=standard_options[0], label="Response")
                    question_components.append(q_label)
                    answer_components.append(q_response)
            
            with gr.Row():
                submit_button = gr.Button("Submit Feedback")
            
            # Results section
            with gr.Accordion("Results", open=False):
                with gr.Row():
                    recommendations = gr.HTML(label="Recommendations")
                
                with gr.Tab("Charts"):
                    with gr.Row():
                        bar_chart = gr.HTML()
                    
                    with gr.Row():
                        radar_chart = gr.HTML()
                
                with gr.Tab("Detailed Report"):
                    detailed_report = gr.HTML()
            
            # Connect event handlers
            load_questionnaire_button.click(
                fn=run_feedback_questionnaire,
                outputs=[
                    questions_state,
                    *answer_components,
                    feedback_status
                ]
            )
            
            questions_state.change(
                fn=update_questions,
                inputs=[questions_state],
                outputs=[*question_components]
            )
            
            submit_button.click(
                fn=process_feedback_submission,
                inputs=[
                    *answer_components,
                    questions_state
                ],
                outputs=[
                    recommendations,
                    feedback_status,
                    bar_chart,
                    radar_chart,
                    detailed_report
                ]
            )
    
    return interface

# Update main block to include the new tab
if __name__ == "__main__":
    print("Starting Emotion Detection System...")
    print(f"Data will be saved to: {DATA_DIR}")
    
    # Check for dlib facial landmark predictor
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.isfile(predictor_path):
        print(f"Warning: {predictor_path} not found!")
        print("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("Extract and place in the same directory as this script")
    
    # Create the base Gradio interface
    app = create_interface()
    
    # Add the wellness report tab
    app = add_wellness_report_tab(app)
    
    # Add the emotion-based psychological feedback form tab
    app = add_feedback_tab(app)
    
    # Launch the application
    app.launch(share=False)

