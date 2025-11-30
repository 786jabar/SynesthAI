"""
SynesthAI ADVANCED - Emotion Detection Engine
Multi-modal emotion analysis with deep learning
"""

import cv2
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO
import logging

logging.getLogger('deepface').setLevel(logging.ERROR)

class AdvancedEmotionDetector:
    """Advanced multi-modal emotion detection"""
    
    def __init__(self):
        """Initialize advanced emotion detection models"""
        print("🧠 Loading Advanced Emotion Detection...")
        
        # DeepFace for facial emotion recognition
        print("   Loading DeepFace models...")
        
        # Pose detection for body language
        print("   Loading YOLO pose detection...")
        self.pose_model = YOLO('yolov8n-pose.pt')
        
        # Emotion history for temporal analysis
        self.emotion_history = []
        self.max_history = 30  # 30 frames
        
        print("✅ Advanced Emotion Detection Ready!")
    
    def detect_emotions(self, frame):
        """
        Advanced emotion analysis with temporal smoothing
        
        Returns:
            dict: {
                'primary_emotion': str,
                'intensity': float (0-1),
                'valence': float (-1 to 1),
                'arousal': float (0-1),
                'all_emotions': dict,
                'confidence': float,
                'emotional_trajectory': str,  # rising, falling, stable
                'body_posture': str
            }
        """
        result = {
            'primary_emotion': 'neutral',
            'intensity': 0.5,
            'valence': 0.0,
            'arousal': 0.5,
            'all_emotions': {},
            'confidence': 0.5,
            'emotional_trajectory': 'stable',
            'body_posture': 'neutral'
        }
        
        try:
            # Analyze with DeepFace
            analysis = DeepFace.analyze(
                frame, 
                actions=['emotion'],
                enforce_detection=False,
                silent=True,
                detector_backend='opencv'
            )
            
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            # Get emotions
            emotions = analysis['emotion']
            result['all_emotions'] = emotions
            
            # Find primary emotion
            primary = max(emotions.items(), key=lambda x: x[1])
            result['primary_emotion'] = primary[0].lower()
            result['intensity'] = primary[1] / 100.0
            result['confidence'] = primary[1] / 100.0
            
            # Calculate valence (positive vs negative)
            result['valence'] = self._calculate_valence(emotions)
            
            # Calculate arousal (energy level)
            result['arousal'] = self._calculate_arousal(emotions)
            
        except Exception as e:
            # Fallback to basic detection
            print(f"DeepFace error: {e}")
            result = self._fallback_detection(frame)
        
        # Analyze body pose
        try:
            pose_results = self.pose_model.predict(frame, save=False, verbose=False)
            if len(pose_results) > 0 and pose_results[0].keypoints is not None:
                keypoints = pose_results[0].keypoints.xyn.numpy()
                if len(keypoints) > 0:
                    result['body_posture'] = self._analyze_posture(keypoints[0])
                    # Adjust arousal based on body language
                    posture_arousal = self._posture_to_arousal(result['body_posture'])
                    result['arousal'] = (result['arousal'] + posture_arousal) / 2
        except:
            pass
        
        # Add to history and analyze trajectory
        self.emotion_history.append({
            'valence': result['valence'],
            'arousal': result['arousal'],
            'emotion': result['primary_emotion']
        })
        
        if len(self.emotion_history) > self.max_history:
            self.emotion_history.pop(0)
        
        # Calculate emotional trajectory
        if len(self.emotion_history) >= 10:
            result['emotional_trajectory'] = self._calculate_trajectory()
        
        # Smooth the values using history
        if len(self.emotion_history) >= 5:
            result = self._smooth_emotions(result)
        
        return result
    
    def _calculate_valence(self, emotions):
        """Calculate emotional valence (positive vs negative)"""
        positive = emotions.get('happy', 0) + emotions.get('surprise', 0) * 0.3
        negative = (emotions.get('sad', 0) + 
                   emotions.get('angry', 0) + 
                   emotions.get('fear', 0) * 0.8 +
                   emotions.get('disgust', 0) * 0.6)
        
        valence = (positive - negative) / 100.0
        return np.clip(valence, -1.0, 1.0)
    
    def _calculate_arousal(self, emotions):
        """Calculate emotional arousal (energy level)"""
        high_arousal = (emotions.get('angry', 0) + 
                       emotions.get('fear', 0) + 
                       emotions.get('surprise', 0) * 0.7 +
                       emotions.get('happy', 0) * 0.5)
        
        low_arousal = emotions.get('sad', 0) + emotions.get('neutral', 0)
        
        total = high_arousal + low_arousal
        if total == 0:
            return 0.5
        
        arousal = high_arousal / total
        return np.clip(arousal, 0.0, 1.0)
    
    def _analyze_posture(self, keypoints):
        """Advanced body posture analysis"""
        if len(keypoints) < 17:
            return 'unknown'
        
        # Get key points
        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_elbow = keypoints[7]
        right_elbow = keypoints[8]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        
        # Calculate shoulder width (openness)
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        
        # Check if arms are raised
        arms_raised = (left_wrist[1] < left_shoulder[1]) or (right_wrist[1] < right_shoulder[1])
        
        # Check if hands are near face (thinking/worried)
        hands_near_face = (abs(left_wrist[1] - nose[1]) < 0.15) or (abs(right_wrist[1] - nose[1]) < 0.15)
        
        # Check if arms are crossed (closed)
        arms_crossed = abs(left_wrist[0] - right_shoulder[0]) < 0.1 or abs(right_wrist[0] - left_shoulder[0]) < 0.1
        
        # Check spine angle (posture)
        try:
            mid_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            mid_hip_y = (left_hip[1] + right_hip[1]) / 2
            upright = (mid_shoulder_y < mid_hip_y - 0.2)
        except:
            upright = True
        
        # Classify posture
        if arms_raised and shoulder_width > 0.3:
            return 'energetic_open'  # Happy, excited
        elif arms_raised:
            return 'energetic_tense'  # Anxious, stressed
        elif hands_near_face:
            return 'contemplative'  # Thinking, worried
        elif arms_crossed:
            return 'defensive'  # Closed off, uncomfortable
        elif shoulder_width > 0.3 and upright:
            return 'confident_open'  # Confident, relaxed
        elif not upright:
            return 'slouched'  # Tired, sad
        else:
            return 'neutral'
    
    def _posture_to_arousal(self, posture):
        """Convert posture to arousal level"""
        posture_arousal_map = {
            'energetic_open': 0.9,
            'energetic_tense': 0.8,
            'contemplative': 0.4,
            'defensive': 0.6,
            'confident_open': 0.6,
            'slouched': 0.2,
            'neutral': 0.5,
            'unknown': 0.5
        }
        return posture_arousal_map.get(posture, 0.5)
    
    def _calculate_trajectory(self):
        """Calculate if emotions are rising, falling, or stable"""
        if len(self.emotion_history) < 10:
            return 'stable'
        
        recent = self.emotion_history[-10:]
        valences = [e['valence'] for e in recent]
        
        # Simple linear regression
        x = np.arange(len(valences))
        slope = np.polyfit(x, valences, 1)[0]
        
        if slope > 0.05:
            return 'rising'  # Getting more positive
        elif slope < -0.05:
            return 'falling'  # Getting more negative
        else:
            return 'stable'
    
    def _smooth_emotions(self, current):
        """Smooth emotions using exponential moving average"""
        if len(self.emotion_history) < 2:
            return current
        
        alpha = 0.3  # Smoothing factor
        
        recent = self.emotion_history[-5:]
        avg_valence = np.mean([e['valence'] for e in recent])
        avg_arousal = np.mean([e['arousal'] for e in recent])
        
        current['valence'] = alpha * current['valence'] + (1 - alpha) * avg_valence
        current['arousal'] = alpha * current['arousal'] + (1 - alpha) * avg_arousal
        
        return current
    
    def _fallback_detection(self, frame):
        """Fallback basic emotion detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            return {
                'primary_emotion': 'neutral',
                'intensity': 0.5,
                'valence': 0.0,
                'arousal': 0.5,
                'all_emotions': {'neutral': 50.0},
                'confidence': 0.5,
                'emotional_trajectory': 'stable',
                'body_posture': 'neutral'
            }
        
        return {
            'primary_emotion': 'neutral',
            'intensity': 0.3,
            'valence': 0.0,
            'arousal': 0.3,
            'all_emotions': {'neutral': 30.0},
            'confidence': 0.3,
            'emotional_trajectory': 'stable',
            'body_posture': 'unknown'
        }
    
    def get_emotion_description(self, emotional_state):
        """Generate human-readable description of emotional state"""
        emotion = emotional_state['primary_emotion']
        valence = emotional_state['valence']
        arousal = emotional_state['arousal']
        trajectory = emotional_state.get('emotional_trajectory', 'stable')
        
        # Energy level
        if arousal > 0.7:
            energy = "highly energetic"
        elif arousal > 0.5:
            energy = "moderately energetic"
        else:
            energy = "calm"
        
        # Positivity
        if valence > 0.3:
            mood = "positive"
        elif valence < -0.3:
            mood = "negative"
        else:
            mood = "neutral"
        
        # Trajectory
        if trajectory == 'rising':
            trend = ", improving"
        elif trajectory == 'falling':
            trend = ", declining"
        else:
            trend = ""
        
        return f"{emotion.title()} - {energy}, {mood}{trend}"


if __name__ == "__main__":
    print("Testing Advanced Emotion Detector...")
    detector = AdvancedEmotionDetector()
    
    cap = cv2.VideoCapture(0)
    print("\n📸 Press 'q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        emotional_state = detector.detect_emotions(frame)
        
        # Display detailed results
        description = detector.get_emotion_description(emotional_state)
        
        cv2.putText(frame, description, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {emotional_state['confidence']:.2f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Posture: {emotional_state['body_posture']}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Advanced Emotion Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Test complete!")
