"""
SynesthAI ADVANCED - Music Generator
Rich procedural music with harmonies and adaptive composition
"""

import numpy as np
import pygame
from threading import Thread
import time

class AdvancedMusicGenerator:
    """Advanced procedural music with harmonies"""
    
    def __init__(self):
        """Initialize advanced music generator"""
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.playing = False
        self.current_emotion = 'neutral'
        self.valence = 0.0
        self.arousal = 0.5
        self.trajectory = 'stable'
        self.thread = None
        
        # Musical scales
        self.scales = self._define_scales()
        self.current_scale = self.scales['major']
        self.root_note = 261.63  # Middle C
        
        # Chord progressions
        self.chord_progressions = {
            'happy': [0, 3, 4, 0],  # I-IV-V-I
            'sad': [0, 2, 3, 0],    # i-III-iv-i
            'peaceful': [0, 3, 0, 4],  # I-IV-I-V
            'tense': [0, 1, 0, 2]    # i-ii-i-III
        }
        
        self.chord_index = 0
        
        print("🎵 Advanced Music Generator Ready!")
    
    def start(self):
        """Start music generation"""
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
        self.trajectory = emotional_state.get('emotional_trajectory', 'stable')
        
        # Update scale based on valence
        if self.valence > 0.2:
            self.current_scale = self.scales['major']
        elif self.valence < -0.2:
            self.current_scale = self.scales['minor']
        else:
            self.current_scale = self.scales['pentatonic']
    
    def _define_scales(self):
        """Define musical scales"""
        return {
            'major': [0, 2, 4, 5, 7, 9, 11],  # Major scale intervals
            'minor': [0, 2, 3, 5, 7, 8, 10],  # Natural minor
            'pentatonic': [0, 2, 4, 7, 9],    # Pentatonic
            'chromatic': list(range(12))       # All notes
        }
    
    def _generation_loop(self):
        """Main music generation loop"""
        while self.playing:
            # Determine tempo based on arousal
            tempo = 0.3 + (1 - self.arousal) * 0.7  # Faster when aroused
            
            # Select chord progression based on emotion
            progression = self._get_chord_progression()
            
            # Play chord with melody
            self._play_musical_phrase(progression, tempo)
            
            # Move to next chord in progression
            self.chord_index = (self.chord_index + 1) % len(progression)
    
    def _get_chord_progression(self):
        """Get appropriate chord progression for current emotion"""
        if self.valence > 0.3:
            return self.chord_progressions['happy']
        elif self.valence < -0.3:
            return self.chord_progressions['sad']
        elif self.arousal < 0.3:
            return self.chord_progressions['peaceful']
        else:
            return self.chord_progressions['tense']
    
    def _play_musical_phrase(self, progression, tempo):
        """Play a musical phrase with chord and melody"""
        # Get current chord root
        chord_root_index = progression[self.chord_index % len(progression)]
        chord_root = self._get_frequency(chord_root_index)
        
        # Generate chord (3 notes)
        chord_freqs = [
            chord_root,
            self._get_frequency(chord_root_index + 2),  # Third
            self._get_frequency(chord_root_index + 4)   # Fifth
        ]
        
        # Create chord sound
        chord_sound = self._generate_chord(chord_freqs, tempo * 0.8)
        
        # Play chord
        channel = pygame.mixer.find_channel()
        if channel:
            channel.play(chord_sound)
        
        # Play melody notes over chord
        num_melody_notes = int(2 + self.arousal * 4)
        for i in range(num_melody_notes):
            # Select melody note from scale
            scale_index = int((i + chord_root_index) % len(self.current_scale))
            melody_freq = self._get_frequency(self.current_scale[scale_index] + 12)  # Octave higher
            
            # Generate melody note
            note_duration = tempo / num_melody_notes
            melody_sound = self._generate_note(melody_freq, note_duration, 'melody')
            
            # Play melody note
            channel = pygame.mixer.find_channel()
            if channel:
                channel.play(melody_sound)
            
            time.sleep(note_duration * 0.8)
    
    def _get_frequency(self, scale_index):
        """Get frequency for scale index"""
        # Ensure index is within scale
        scale_index = scale_index % len(self.current_scale)
        semitone = self.current_scale[scale_index]
        
        # Calculate frequency (equal temperament)
        return self.root_note * (2 ** (semitone / 12))
    
    def _generate_chord(self, frequencies, duration):
        """Generate a chord (multiple notes together)"""
        sample_rate = 44100
        num_samples = int(sample_rate * duration)
        
        t = np.linspace(0, duration, num_samples, False)
        
        # Combine multiple frequencies
        wave = np.zeros(num_samples)
        for freq in frequencies:
            # Sine wave with harmonics
            wave += np.sin(2 * np.pi * freq * t) * 0.3
            wave += np.sin(4 * np.pi * freq * t) * 0.15  # Second harmonic
            wave += np.sin(6 * np.pi * freq * t) * 0.08  # Third harmonic
        
        # ADSR envelope
        envelope = self._create_envelope(num_samples, 'chord')
        wave = wave * envelope
        
        # Normalize and convert
        wave = wave / np.max(np.abs(wave)) * 0.2
        wave = (wave * 32767).astype(np.int16)
        
        # Stereo
        stereo_wave = np.column_stack((wave, wave))
        
        return pygame.sndarray.make_sound(stereo_wave)
    
    def _generate_note(self, frequency, duration, note_type='melody'):
        """Generate a single note with rich harmonics"""
        sample_rate = 44100
        num_samples = int(sample_rate * duration)
        
        t = np.linspace(0, duration, num_samples, False)
        
        # Rich waveform with multiple harmonics
        wave = np.sin(2 * np.pi * frequency * t)  # Fundamental
        wave += 0.5 * np.sin(4 * np.pi * frequency * t)  # 2nd harmonic
        wave += 0.3 * np.sin(6 * np.pi * frequency * t)  # 3rd harmonic
        wave += 0.2 * np.sin(8 * np.pi * frequency * t)  # 4th harmonic
        wave += 0.1 * np.sin(10 * np.pi * frequency * t)  # 5th harmonic
        
        # Add slight detuning for warmth
        detune = frequency * 1.005
        wave += 0.3 * np.sin(2 * np.pi * detune * t)
        
        # ADSR envelope
        envelope = self._create_envelope(num_samples, note_type)
        wave = wave * envelope
        
        # Normalize
        wave = wave / np.max(np.abs(wave)) * 0.25
        wave = (wave * 32767).astype(np.int16)
        
        # Stereo
        stereo_wave = np.column_stack((wave, wave))
        
        return pygame.sndarray.make_sound(stereo_wave)
    
    def _create_envelope(self, num_samples, envelope_type='melody'):
        """Create ADSR envelope"""
        if envelope_type == 'chord':
            # Longer, sustained envelope for chords
            attack = int(num_samples * 0.15)
            decay = int(num_samples * 0.25)
            sustain_level = 0.6
            sustain = int(num_samples * 0.45)
            release = num_samples - attack - decay - sustain
        else:
            # Shorter, more percussive for melody
            attack = int(num_samples * 0.05)
            decay = int(num_samples * 0.15)
            sustain_level = 0.7
            sustain = int(num_samples * 0.6)
            release = num_samples - attack - decay - sustain
        
        # Create envelope segments
        attack_env = np.linspace(0, 1, attack)
        decay_env = np.linspace(1, sustain_level, decay)
        sustain_env = np.ones(sustain) * sustain_level
        release_env = np.linspace(sustain_level, 0, release)
        
        envelope = np.concatenate([attack_env, decay_env, sustain_env, release_env])
        
        # Ensure correct length
        if len(envelope) != num_samples:
            envelope = np.resize(envelope, num_samples)
        
        return envelope


if __name__ == "__main__":
    print("Testing Advanced Music Generator...")
    print("Playing emotional soundscapes...")
    print("Press Ctrl+C to stop\n")
    
    generator = AdvancedMusicGenerator()
    generator.start()
    
    # Test different emotions
    emotions = [
        {'primary_emotion': 'happy', 'valence': 0.8, 'arousal': 0.7, 'emotional_trajectory': 'rising'},
        {'primary_emotion': 'sad', 'valence': -0.6, 'arousal': 0.3, 'emotional_trajectory': 'stable'},
        {'primary_emotion': 'neutral', 'valence': 0.0, 'arousal': 0.5, 'emotional_trajectory': 'stable'},
    ]
    
    try:
        for emotion in emotions:
            print(f"🎵 Playing: {emotion['primary_emotion']}")
            generator.update_emotion(emotion)
            time.sleep(8)
    
    except KeyboardInterrupt:
        pass
    
    finally:
        generator.stop()
        print("\n✅ Test complete!")
