"""
SynesthAI - Music Generator  
Generates simple procedural music based on emotions
"""

import numpy as np
import pygame
from threading import Thread
import time

class MusicGenerator:
    """Simple procedural music generator"""
    
    def __init__(self):
        """Initialize music generator"""
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        self.playing = False
        self.current_emotion = 'neutral'
        self.thread = None
        
        print("🎵 Music Generator Ready!")
    
    def start(self):
        """Start music generation thread"""
        if not self.playing:
            self.playing = True
            self.thread = Thread(target=self._generation_loop, daemon=True)
            self.thread.start()
    
    def stop(self):
        """Stop music generation"""
        self.playing = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def update_emotion(self, emotional_state):
        """Update music based on emotional state"""
        self.current_emotion = emotional_state['primary_emotion']
        self.valence = emotional_state.get('valence', 0)
        self.arousal = emotional_state.get('arousal', 0.5)
    
    def _generation_loop(self):
        """Main music generation loop"""
        while self.playing:
            # Generate and play a note based on current emotion
            note_freq = self._emotion_to_frequency(self.current_emotion)
            duration = 0.3 + (1 - self.arousal) * 0.5  # Slower when calm
            
            # Generate sound
            sound = self._generate_note(note_freq, duration)
            
            # Play
            channel = pygame.mixer.find_channel()
            if channel:
                channel.play(sound)
            
            # Wait before next note
            time.sleep(duration * 0.8)
    
    def _emotion_to_frequency(self, emotion):
        """Map emotion to musical note frequency"""
        # Musical scales (frequencies in Hz)
        # Major scale for positive emotions
        major_scale = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]  # C major
        
        # Minor scale for negative emotions  
        minor_scale = [220.00, 246.94, 261.63, 293.66, 329.63, 349.23, 392.00]  # A minor
        
        # Map emotions to scales
        if emotion in ['happy', 'surprise']:
            scale = major_scale
            index = int((self.valence + 1) * 3)  # Higher notes for positive
        elif emotion in ['sad', 'fear']:
            scale = minor_scale
            index = int((1 - abs(self.valence)) * 3)  # Lower notes for negative
        elif emotion == 'angry':
            scale = minor_scale
            index = 5 + int(self.arousal * 2)  # Higher, tense notes
        else:  # neutral
            scale = major_scale
            index = 3  # Middle note
        
        index = np.clip(index, 0, len(scale) - 1)
        return scale[index]
    
    def _generate_note(self, frequency, duration):
        """Generate a musical note"""
        sample_rate = 22050
        num_samples = int(sample_rate * duration)
        
        # Time array
        t = np.linspace(0, duration, num_samples, False)
        
        # Generate sine wave with envelope
        wave = np.sin(2 * np.pi * frequency * t)
        
        # ADSR envelope (simplified)
        attack = int(num_samples * 0.1)
        decay = int(num_samples * 0.2)
        sustain = int(num_samples * 0.5)
        release = num_samples - attack - decay - sustain
        
        envelope = np.concatenate([
            np.linspace(0, 1, attack),
            np.linspace(1, 0.7, decay),
            np.ones(sustain) * 0.7,
            np.linspace(0.7, 0, release)
        ])
        
        wave = wave * envelope
        
        # Add some harmonics for richer sound
        wave += 0.3 * np.sin(4 * np.pi * frequency * t) * envelope
        wave += 0.2 * np.sin(6 * np.pi * frequency * t) * envelope
        
        # Normalize and convert to pygame sound
        wave = wave * 0.3  # Volume
        wave = (wave * 32767).astype(np.int16)
        
        # Stereo
        stereo_wave = np.column_stack((wave, wave))
        
        return pygame.sndarray.make_sound(stereo_wave)


if __name__ == "__main__":
    # Test the music generator
    print("Testing Music Generator...")
    print("Playing different emotional tones...")
    print("Press Ctrl+C to stop\n")
    
    generator = MusicGenerator()
    generator.start()
    
    # Test different emotions
    emotions = [
        {'primary_emotion': 'happy', 'valence': 0.8, 'arousal': 0.7},
        {'primary_emotion': 'sad', 'valence': -0.6, 'arousal': 0.3},
        {'primary_emotion': 'angry', 'valence': -0.5, 'arousal': 0.9},
        {'primary_emotion': 'neutral', 'valence': 0.0, 'arousal': 0.5},
    ]
    
    try:
        for emotion in emotions:
            print(f"🎵 Playing: {emotion['primary_emotion']}")
            generator.update_emotion(emotion)
            time.sleep(5)
    
    except KeyboardInterrupt:
        pass
    
    finally:
        generator.stop()
        print("\n✅ Test complete!")
