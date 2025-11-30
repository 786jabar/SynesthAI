"""
SynesthAI ADVANCED - Visual Art Generator
Multi-layered generative art with advanced effects
"""

import numpy as np
import cv2
from colorsys import hsv_to_rgb, rgb_to_hsv
import math

class AdvancedVisualArtGenerator:
    """Advanced generative art with multiple visual layers"""
    
    def __init__(self, width=1200, height=800):
        """Initialize advanced art generator"""
        self.width = width
        self.height = height
        
        # Multiple canvas layers
        self.background_layer = np.zeros((height, width, 3), dtype=np.uint8)
        self.particle_layer = np.zeros((height, width, 3), dtype=np.uint8)
        self.effect_layer = np.zeros((height, width, 3), dtype=np.uint8)
        self.final_canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Particle system
        self.particles = []
        self.max_particles = 500
        
        # Wave/flow field
        self.flow_field = self._generate_flow_field()
        self.flow_time = 0
        
        # Color palette history
        self.color_history = []
        
        print("🎨 Advanced Visual Art Generator Ready!")
    
    def generate_frame(self, emotional_state):
        """
        Generate advanced artistic frame
        
        Args:
            emotional_state: dict from AdvancedEmotionDetector
        """
        # Get emotion parameters
        emotion = emotional_state['primary_emotion']
        valence = emotional_state['valence']
        arousal = emotional_state['arousal']
        intensity = emotional_state['intensity']
        trajectory = emotional_state.get('emotional_trajectory', 'stable')
        
        # Update flow field based on emotion
        self._update_flow_field(arousal, valence)
        
        # Generate dynamic background gradient
        self._generate_background(emotion, valence, arousal)
        
        # Create particles
        particle_count = int(5 + arousal * 15)
        for _ in range(particle_count):
            self._create_particle(emotion, valence, arousal, intensity)
        
        # Update and render particles
        self._update_particles(valence, arousal, trajectory)
        
        # Add special effects based on emotion
        self._add_emotion_effects(emotion, intensity, trajectory)
        
        # Combine layers
        self._composite_layers()
        
        return self.final_canvas.copy()
    
    def _generate_flow_field(self):
        """Generate Perlin-like flow field"""
        field = np.zeros((self.height // 20, self.width // 20, 2))
        
        for y in range(field.shape[0]):
            for x in range(field.shape[1]):
                angle = np.sin(x * 0.1) * np.cos(y * 0.1) * np.pi * 2
                field[y, x] = [np.cos(angle), np.sin(angle)]
        
        return field
    
    def _update_flow_field(self, arousal, valence):
        """Update flow field dynamically"""
        self.flow_time += 0.02 * (1 + arousal)
        
        for y in range(self.flow_field.shape[0]):
            for x in range(self.flow_field.shape[1]):
                angle = (np.sin(x * 0.1 + self.flow_time) * 
                        np.cos(y * 0.1 + self.flow_time) * np.pi * 2)
                angle += valence * np.pi  # Rotate based on valence
                
                self.flow_field[y, x] = [np.cos(angle), np.sin(angle)]
    
    def _generate_background(self, emotion, valence, arousal):
        """Generate dynamic gradient background"""
        # Get primary and secondary colors
        primary_hue, sat, bright = self._emotion_to_color(emotion, valence, arousal)
        secondary_hue = (primary_hue + 60) % 360  # Complementary color
        
        # Create gradient
        for y in range(self.height):
            t = y / self.height
            
            # Interpolate between colors
            hue = primary_hue + (secondary_hue - primary_hue) * t
            s = sat * (0.5 + 0.5 * arousal)
            b = bright * (0.3 + 0.4 * (1 - t))
            
            r, g, b_val = hsv_to_rgb(hue / 360, s, b)
            color = (int(b_val * 255), int(g * 255), int(r * 255))
            
            cv2.line(self.background_layer, (0, y), (self.width, y), color, 1)
        
        # Add noise/texture
        noise = np.random.randint(-20, 20, (self.height, self.width, 3), dtype=np.int16)
        self.background_layer = np.clip(self.background_layer.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    def _emotion_to_color(self, emotion, valence, arousal):
        """Advanced emotion to color mapping"""
        emotion_colors = {
            'happy': (45, 0.85, 0.95),      # Bright yellow
            'sad': (210, 0.75, 0.50),       # Deep blue
            'angry': (0, 0.95, 0.85),       # Intense red
            'fear': (270, 0.70, 0.60),      # Purple
            'surprise': (180, 0.80, 0.90),  # Cyan
            'disgust': (120, 0.70, 0.65),   # Green
            'neutral': (200, 0.30, 0.70)    # Light blue-gray
        }
        
        base_hue, base_sat, base_bright = emotion_colors.get(emotion, (0, 0.5, 0.5))
        
        # Adjust based on valence and arousal
        hue = base_hue
        sat = base_sat * (0.5 + arousal * 0.5)
        bright = base_bright * (0.5 + (valence + 1) * 0.25)
        
        return hue, sat, bright
    
    def _create_particle(self, emotion, valence, arousal, intensity):
        """Create advanced particle with flow field influence"""
        if len(self.particles) >= self.max_particles:
            self.particles.pop(0)
        
        # Random starting position
        x = np.random.randint(0, self.width)
        y = np.random.randint(0, self.height)
        
        # Get base velocity from flow field
        fx = int(x / 20) % self.flow_field.shape[1]
        fy = int(y / 20) % self.flow_field.shape[0]
        flow = self.flow_field[fy, fx]
        
        speed = 2 + arousal * 6
        vx = flow[0] * speed + np.random.randn() * 0.5
        vy = flow[1] * speed + np.random.randn() * 0.5
        
        # Color based on emotion
        hue, sat, bright = self._emotion_to_color(emotion, valence, arousal)
        
        # Add variation
        hue = (hue + np.random.randint(-30, 30)) % 360
        
        r, g, b = hsv_to_rgb(hue / 360, sat, bright)
        color = (int(b * 255), int(g * 255), int(r * 255))
        
        # Size based on intensity
        size = int(3 + intensity * 12)
        
        particle = {
            'x': float(x),
            'y': float(y),
            'vx': vx,
            'vy': vy,
            'size': size,
            'color': color,
            'life': 1.0,
            'trail': []  # For motion trails
        }
        
        self.particles.append(particle)
    
    def _update_particles(self, valence, arousal, trajectory):
        """Update particle positions with advanced physics"""
        particles_to_remove = []
        
        for i, particle in enumerate(self.particles):
            # Store trail
            if len(particle['trail']) > 10:
                particle['trail'].pop(0)
            particle['trail'].append((int(particle['x']), int(particle['y'])))
            
            # Get flow field influence
            fx = int(particle['x'] / 20) % self.flow_field.shape[1]
            fy = int(particle['y'] / 20) % self.flow_field.shape[0]
            flow = self.flow_field[fy, fx]
            
            # Apply flow field
            particle['vx'] += flow[0] * 0.3
            particle['vy'] += flow[1] * 0.3
            
            # Apply gravity based on valence
            particle['vy'] -= valence * 0.4  # Negative valence = fall
            
            # Apply drag
            drag = 0.98 if arousal > 0.5 else 0.95
            particle['vx'] *= drag
            particle['vy'] *= drag
            
            # Update position
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            
            # Bounce off edges with energy loss
            if particle['x'] < 0 or particle['x'] >= self.width:
                particle['vx'] *= -0.7
                particle['x'] = np.clip(particle['x'], 0, self.width - 1)
            
            if particle['y'] < 0 or particle['y'] >= self.height:
                particle['vy'] *= -0.7
                particle['y'] = np.clip(particle['y'], 0, self.height - 1)
            
            # Decay life
            particle['life'] *= 0.98
            
            # Draw particle with glow effect
            if particle['life'] > 0.01:
                self._draw_particle_with_glow(particle)
            else:
                particles_to_remove.append(i)
        
        # Remove dead particles
        for i in reversed(particles_to_remove):
            self.particles.pop(i)
    
    def _draw_particle_with_glow(self, particle):
        """Draw particle with glow effect"""
        x, y = int(particle['x']), int(particle['y'])
        size = particle['size']
        alpha = particle['life']
        
        # Draw motion trail
        if len(particle['trail']) > 1:
            for j in range(len(particle['trail']) - 1):
                trail_alpha = alpha * (j / len(particle['trail']))
                color = tuple(int(c * trail_alpha) for c in particle['color'])
                cv2.line(self.particle_layer, particle['trail'][j], 
                        particle['trail'][j + 1], color, 1)
        
        # Draw glow (outer)
        glow_color = tuple(int(c * alpha * 0.3) for c in particle['color'])
        cv2.circle(self.particle_layer, (x, y), size + 3, glow_color, -1, cv2.LINE_AA)
        
        # Draw core (inner)
        core_color = tuple(int(c * alpha) for c in particle['color'])
        cv2.circle(self.particle_layer, (x, y), size, core_color, -1, cv2.LINE_AA)
    
    def _add_emotion_effects(self, emotion, intensity, trajectory):
        """Add special effects based on emotion"""
        # Reset effect layer
        self.effect_layer = np.zeros_like(self.effect_layer)
        
        if emotion == 'happy' and intensity > 0.6:
            # Add sparkles
            for _ in range(int(intensity * 20)):
                x = np.random.randint(0, self.width)
                y = np.random.randint(0, self.height)
                size = np.random.randint(2, 6)
                cv2.circle(self.effect_layer, (x, y), size, (255, 255, 200), -1)
        
        elif emotion == 'sad' and intensity > 0.5:
            # Add rain effect
            for _ in range(int(intensity * 30)):
                x = np.random.randint(0, self.width)
                y = np.random.randint(0, self.height)
                cv2.line(self.effect_layer, (x, y), (x, y + 10), (100, 120, 180), 1)
        
        elif emotion == 'angry' and intensity > 0.6:
            # Add sharp lines/cracks
            for _ in range(int(intensity * 15)):
                x1 = np.random.randint(0, self.width)
                y1 = np.random.randint(0, self.height)
                angle = np.random.uniform(0, 2 * np.pi)
                length = np.random.randint(20, 100)
                x2 = int(x1 + length * np.cos(angle))
                y2 = int(y1 + length * np.sin(angle))
                cv2.line(self.effect_layer, (x1, y1), (x2, y2), (200, 50, 50), 2)
    
    def _composite_layers(self):
        """Combine all layers with blending"""
        # Start with background
        self.final_canvas = self.background_layer.copy()
        
        # Add particles with blending
        particle_alpha = 0.9
        self.final_canvas = cv2.addWeighted(self.final_canvas, 1.0, 
                                           self.particle_layer, particle_alpha, 0)
        
        # Add effects
        effect_alpha = 0.5
        self.final_canvas = cv2.addWeighted(self.final_canvas, 1.0,
                                           self.effect_layer, effect_alpha, 0)
        
        # Apply slight blur for dreamlike quality
        self.final_canvas = cv2.GaussianBlur(self.final_canvas, (3, 3), 0)
        
        # Reset particle layer for next frame (with fade)
        self.particle_layer = cv2.addWeighted(self.particle_layer, 0.85, 
                                             np.zeros_like(self.particle_layer), 0.15, 0)
