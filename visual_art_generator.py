"""
SynesthAI - Visual Art Generator
Creates real-time generative art based on emotional state
"""

import numpy as np
import cv2
from colorsys import hsv_to_rgb

class VisualArtGenerator:
    """Generates dynamic visual art based on emotions"""
    
    def __init__(self, width=800, height=600):
        """Initialize art generator"""
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.particles = []
        self.max_particles = 200
        
        print("🎨 Visual Art Generator Ready!")
    
    def generate_frame(self, emotional_state):
        """
        Generate one frame of art based on emotional state
        
        Args:
            emotional_state: dict from EmotionDetector
            
        Returns:
            numpy array: RGB image
        """
        # Fade previous frame for trail effect
        self.canvas = cv2.addWeighted(self.canvas, 0.92, np.zeros_like(self.canvas), 0.08, 0)
        
        # Get emotion-based parameters
        emotion = emotional_state['primary_emotion']
        valence = emotional_state['valence']
        arousal = emotional_state['arousal']
        intensity = emotional_state['intensity']
        
        # Map emotion to visual parameters
        particle_speed = 1 + arousal * 4
        particle_size = int(2 + intensity * 8)
        num_new_particles = int(1 + arousal * 5)
        
        # Get color from emotion
        hue, saturation, brightness = self._emotion_to_color(emotional_state)
        
        # Create new particles
        for _ in range(num_new_particles):
            self._add_particle(hue, saturation, brightness, particle_speed, particle_size)
        
        # Update and draw particles
        self._update_particles(valence)
        
        return self.canvas.copy()
    
    def _emotion_to_color(self, emotional_state):
        """Convert emotional state to HSV color"""
        emotion = emotional_state['primary_emotion']
        valence = emotional_state['valence']
        arousal = emotional_state['arousal']
        
        # Base colors for emotions (0-360 hue)
        emotion_hues = {
            'happy': 45,      # Warm yellow/orange
            'sad': 210,       # Deep blue
            'angry': 0,       # Red
            'fear': 270,      # Purple
            'surprise': 180,  # Cyan
            'disgust': 120,   # Green
            'neutral': 0      # Gray
        }
        
        hue = emotion_hues.get(emotion, 0)
        
        # Adjust saturation based on arousal
        saturation = 0.3 + arousal * 0.7
        
        # Adjust brightness based on valence
        brightness = 0.4 + ((valence + 1) / 2) * 0.6
        
        return hue, saturation, brightness
    
    def _add_particle(self, hue, saturation, brightness, speed, size):
        """Add a new particle to the system"""
        if len(self.particles) >= self.max_particles:
            self.particles.pop(0)
        
        # Random starting position
        x = np.random.randint(0, self.width)
        y = np.random.randint(0, self.height)
        
        # Random velocity based on speed
        angle = np.random.uniform(0, 2 * np.pi)
        vx = np.cos(angle) * speed
        vy = np.sin(angle) * speed
        
        # Convert HSV to RGB
        r, g, b = hsv_to_rgb(hue / 360, saturation, brightness)
        color = (int(b * 255), int(g * 255), int(r * 255))  # BGR for OpenCV
        
        particle = {
            'x': x,
            'y': y,
            'vx': vx,
            'vy': vy,
            'size': size,
            'color': color,
            'life': 1.0
        }
        
        self.particles.append(particle)
    
    def _update_particles(self, valence):
        """Update particle positions and draw them"""
        particles_to_remove = []
        
        for i, particle in enumerate(self.particles):
            # Update position
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            
            # Apply gravity based on valence
            # Negative valence = things fall (sad)
            # Positive valence = things rise (happy)
            particle['vy'] -= valence * 0.3
            
            # Bounce off edges
            if particle['x'] < 0 or particle['x'] >= self.width:
                particle['vx'] *= -0.8
                particle['x'] = np.clip(particle['x'], 0, self.width - 1)
            
            if particle['y'] < 0 or particle['y'] >= self.height:
                particle['vy'] *= -0.8
                particle['y'] = np.clip(particle['y'], 0, self.height - 1)
            
            # Decay life
            particle['life'] *= 0.99
            
            # Draw particle
            if particle['life'] > 0.01:
                alpha = particle['life']
                color = tuple(int(c * alpha) for c in particle['color'])
                
                cv2.circle(
                    self.canvas,
                    (int(particle['x']), int(particle['y'])),
                    particle['size'],
                    color,
                    -1
                )
            else:
                particles_to_remove.append(i)
        
        # Remove dead particles
        for i in reversed(particles_to_remove):
            self.particles.pop(i)
    
    def add_emotion_shape(self, emotional_state):
        """Add a special shape based on current emotion"""
        emotion = emotional_state['primary_emotion']
        intensity = emotional_state['intensity']
        
        hue, sat, bright = self._emotion_to_color(emotional_state)
        r, g, b = hsv_to_rgb(hue / 360, sat, bright)
        color = (int(b * 255), int(g * 255), int(r * 255))
        
        center_x = self.width // 2
        center_y = self.height // 2
        radius = int(50 + intensity * 100)
        
        if emotion == 'happy':
            # Star burst
            pts = []
            for i in range(10):
                angle = i * np.pi / 5
                r_pt = radius if i % 2 == 0 else radius // 2
                x = int(center_x + r_pt * np.cos(angle))
                y = int(center_y + r_pt * np.sin(angle))
                pts.append([x, y])
            pts = np.array(pts)
            cv2.fillPoly(self.canvas, [pts], color)
        
        elif emotion == 'sad':
            # Tear drop
            cv2.ellipse(self.canvas, (center_x, center_y), 
                       (radius // 2, radius), 0, 0, 360, color, -1)
        
        elif emotion == 'angry':
            # Jagged shape
            pts = []
            for i in range(8):
                angle = i * np.pi / 4
                r_pt = radius + np.random.randint(-20, 20)
                x = int(center_x + r_pt * np.cos(angle))
                y = int(center_y + r_pt * np.sin(angle))
                pts.append([x, y])
            pts = np.array(pts)
            cv2.fillPoly(self.canvas, [pts], color)
        
        else:
            # Circle for neutral/other
            cv2.circle(self.canvas, (center_x, center_y), radius, color, -1)


if __name__ == "__main__":
    # Test the art generator
    print("Testing Visual Art Generator...")
    
    generator = VisualArtGenerator()
    
    # Simulate different emotions
    test_emotions = [
        {'primary_emotion': 'happy', 'valence': 0.8, 'arousal': 0.7, 'intensity': 0.9},
        {'primary_emotion': 'sad', 'valence': -0.6, 'arousal': 0.3, 'intensity': 0.7},
        {'primary_emotion': 'angry', 'valence': -0.5, 'arousal': 0.9, 'intensity': 0.8},
        {'primary_emotion': 'neutral', 'valence': 0.0, 'arousal': 0.5, 'intensity': 0.5},
    ]
    
    print("\n🎨 Generating art for different emotions...")
    print("Press any key to cycle through emotions, 'q' to quit\n")
    
    emotion_idx = 0
    
    while True:
        # Generate frame
        frame = generator.generate_frame(test_emotions[emotion_idx])
        
        # Add text
        emotion_name = test_emotions[emotion_idx]['primary_emotion']
        cv2.putText(frame, f"Emotion: {emotion_name.upper()}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('Visual Art Test', frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key != 255:  # Any key pressed
            emotion_idx = (emotion_idx + 1) % len(test_emotions)
            print(f"Switched to: {test_emotions[emotion_idx]['primary_emotion']}")
    
    cv2.destroyAllWindows()
    print("\n✅ Test complete!")
