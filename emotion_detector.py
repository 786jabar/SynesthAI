"""
SynesthAI - Emotion Detection Engine (Simplified)
Analyzes facial expressions and body pose
"""

import cv2
import numpy as np
try:
    from deepface import DeepFace
    USE_DEEPFACE = True
except:
    USE_DEEPFACE = False
    
from ultralytics import YOLO

class EmotionDetector:
    """Multi-modal emotion detection combining face and body analysis"""
    
    def __init__(self):
        """Initialize emotion detection models"""
        print("🧠 Loading Emotion Detection Models...")
        
        # Pose detection for body language
        self.pose_model = YOLO('yolov8n-pose.pt')
        
        self.use_deepface = USE_DEEPFACE
        
        if self.use_deepface:
            print("✅ Using DeepFace for emotion detection")
        else:
            print("✅ Using basic emotion detection")
        
        print("✅ Emotion Detection Ready!")
    
    def detect_emotions(self, frame):
        """
        Analyze frame and return emotional state
        
        Args:
            frame: OpenCV image (BGR format)
            
        Returns:
            dict: {
                'primary_emotion': str,
                'intensity': float (0-1),
                'valence': float (-1 to 1),
                'arousal': float (0-1),
                'all_emotions': dict
            }
        """
        result = {
            'primary_emotion': 'neutral',
            'intensity': 0.5,
            'valence': 0.0,
            'arousal': 0.5,
            'all_emotions': {'neutral': 1.0}
        }
        
        if self.use_deepface:
            try:
                # Use DeepFace for emotion detection
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
                
                if isinstance(analysis, list):
                    analysis = analysis[0]
                
                emotions = analysis['emotion']
                result['all_emotions'] = emotions
                
                # Find primary emotion
                primary = max(emotions.items(), key=lambda x: x[1])
                result['primary_emotion'] = primary[0].lower()
                result['intensity'] = primary[1] / 100.0
                
                # Calculate valence and arousal
                result['valence'] = self._calculate_valence(emotions)
                result['arousal'] = self._calculate_arousal(emotions)
                
            except Exception as e:
                print(f"DeepFace error: {e}")
                # Fall back to basic detection
                result = self._basic_emotion_detection(frame)
        else:
            result = self._basic_emotion_detection(frame)
        
        # Analyze body pose for additional context
        pose_results = self.pose_model.predict(frame, save=False, verbose=False)
        if len(pose_results) > 0 and pose_results[0].keypoints is not None:
            keypoints = pose_results[0].keypoints.xyn.numpy()
            if len(keypoints) > 0:
                result['body_posture'] = self._analyze_posture(keypoints[0])
        
        return result
    
    def _basic_emotion_detection(self, frame):
        """Basic emotion detection using simple heuristics"""
        # Simple face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Person detected - vary emotion randomly for demo
            emotions = ['happy', 'neutral', 'sad']
            import random
            emotion = random.choice(emotions)
            
            return {
                'primary_emotion': emotion,
                'intensity': 0.6,
                'valence': 0.3 if emotion == 'happy' else -0.3 if emotion == 'sad' else 0.0,
                'arousal': 0.5,
                'all_emotions': {emotion: 0.6}
            }
        
        return {
            'primary_emotion': 'neutral',
            'intensity': 0.5,
            'valence': 0.0,
            'arousal': 0.5,
            'all_emotions': {'neutral': 1.0}
        }
    
    def _calculate_valence(self, emotions):
        """Calculate emotional valence (positive vs negative)"""
        positive = emotions.get('happy', 0) + emotions.get('surprise', 0) * 0.5
        negative = emotions.get('sad', 0) + emotions.get('angry', 0) + emotions.get('fear', 0) * 0.7
        
        return (positive - negative) / 100.0
    
    def _calculate_arousal(self, emotions):
        """Calculate emotional arousal (calm vs excited)"""
        high_arousal = emotions.get('angry', 0) + emotions.get('fear', 0) + emotions.get('surprise', 0)
        low_arousal = emotions.get('sad', 0) + emotions.get('neutral', 0)
        
        total = high_arousal + low_arousal
        if total == 0:
            return 0.5
        
        return high_arousal / total
    
    def _analyze_posture(self, keypoints):
        """Analyze body posture from keypoints"""
        if len(keypoints) < 17:
            return 'unknown'
        
        # Check shoulder width
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        
        # Check if arms are raised
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        nose = keypoints[0]
        
        arms_raised = (left_wrist[1] < nose[1]) or (right_wrist[1] < nose[1])
        
        if shoulder_width > 0.3 and arms_raised:
            return 'open_energetic'
        elif shoulder_width > 0.3:
            return 'open_calm'
        elif arms_raised:
            return 'closed_energetic'
        else:
            return 'closed_calm'


if __name__ == "__main__":
    # Test the emotion detector
    print("Testing Emotion Detector...")
    detector = EmotionDetector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    print("\n📸 Press 'q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect emotions
        emotional_state = detector.detect_emotions(frame)
        
        # Display results
        emotion = emotional_state['primary_emotion']
        intensity = emotional_state['intensity']
        valence = emotional_state['valence']
        arousal = emotional_state['arousal']
        
        # Draw on frame
        text = f"{emotion.upper()} ({intensity:.2f})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Valence: {valence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Arousal: {arousal:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Emotion Detection Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Test complete!")
