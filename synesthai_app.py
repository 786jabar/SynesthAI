"""
SynesthAI - Main Application
The AI That Turns Emotions Into Living Art

Run with: streamlit run synesthai_app.py
"""

import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import time

# Import our modules
from emotion_detector import EmotionDetector
from visual_art_generator import VisualArtGenerator
from music_generator import MusicGenerator

# Page configuration
st.set_page_config(
    page_title="SynesthAI",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 3.5em;
        font-weight: 900;
        background: linear-gradient(135deg, #8B5CF6, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 1.3em;
        color: #666;
        margin-bottom: 30px;
    }
    .emotion-box {
        background: linear-gradient(135deg, #667EEA, #764BA2);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stat-box {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">SynesthAI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">The AI That Turns Your Emotions Into Living Art</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    session_mode = st.selectbox(
        "Session Mode",
        ["Live Experience", "Demo Mode", "Save Recording"]
    )
    
    st.markdown("---")
    
    enable_music = st.checkbox("🎵 Enable Music", value=True)
    show_emotion_data = st.checkbox("📊 Show Emotion Data", value=True)
    show_particles = st.checkbox("✨ Show Particles", value=True)
    
    st.markdown("---")
    
    st.header("📖 About")
    st.write("""
    **SynesthAI** reads your emotions in real-time and creates a unique multi-sensory artistic experience.
    
    - 🧠 Emotion Detection
    - 🎨 Generative Art  
    - 🎵 Procedural Music
    - 💫 Adaptive Response
    """)
    
    st.markdown("---")
    st.caption("MD Jawar Safi • AI for Creativity")

# Initialize session state
if 'emotion_detector' not in st.session_state:
    with st.spinner("🧠 Loading AI models..."):
        st.session_state.emotion_detector = EmotionDetector()
        st.session_state.art_generator = VisualArtGenerator(width=800, height=600)
        st.session_state.music_generator = MusicGenerator()
    st.success("✅ AI Models Loaded!")

if 'running' not in st.session_state:
    st.session_state.running = False

if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎨 Live Emotional Art")
    video_placeholder = st.empty()
    art_placeholder = st.empty()

with col2:
    st.subheader("😊 Emotional State")
    emotion_placeholder = st.empty()
    
    if show_emotion_data:
        st.subheader("📊 Emotion Analytics")
        chart_placeholder = st.empty()

# Control buttons
button_col1, button_col2, button_col3 = st.columns(3)

with button_col1:
    if st.button("▶️ Start Experience", use_container_width=True):
        st.session_state.running = True
        if enable_music:
            st.session_state.music_generator.start()

with button_col2:
    if st.button("⏸️ Pause", use_container_width=True):
        st.session_state.running = False
        st.session_state.music_generator.stop()

with button_col3:
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.art_generator = VisualArtGenerator(width=800, height=600)
        st.session_state.emotion_history = []
        st.session_state.running = False
        st.session_state.music_generator.stop()

# Main loop
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Could not open webcam. Please check your camera.")
    else:
        frame_count = 0
        
        while st.session_state.running:
            ret, frame = cap.read()
            
            if not ret:
                st.error("❌ Could not read frame from webcam")
                break
            
            frame_count += 1
            
            # Process every 3rd frame to reduce load
            if frame_count % 3 == 0:
                # Detect emotions
                emotional_state = st.session_state.emotion_detector.detect_emotions(frame)
                
                # Update music
                if enable_music:
                    st.session_state.music_generator.update_emotion(emotional_state)
                
                # Generate art
                if show_particles:
                    art_frame = st.session_state.art_generator.generate_frame(emotional_state)
                else:
                    art_frame = np.zeros((600, 800, 3), dtype=np.uint8)
                    st.session_state.art_generator.add_emotion_shape(emotional_state)
                    art_frame = st.session_state.art_generator.canvas.copy()
                
                # Save to history
                st.session_state.emotion_history.append({
                    'time': datetime.now(),
                    'emotion': emotional_state['primary_emotion'],
                    'valence': emotional_state['valence'],
                    'arousal': emotional_state['arousal']
                })
                
                # Keep only last 100 points
                if len(st.session_state.emotion_history) > 100:
                    st.session_state.emotion_history.pop(0)
                
                # Display video feed
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Display art
                art_rgb = cv2.cvtColor(art_frame, cv2.COLOR_BGR2RGB)
                art_placeholder.image(art_rgb, channels="RGB", use_column_width=True)
                
                # Display emotion info
                emotion_html = f"""
                <div class="emotion-box">
                    <h2 style="margin: 0;">{emotional_state['primary_emotion'].upper()}</h2>
                    <p style="font-size: 1.2em; margin: 10px 0;">Intensity: {emotional_state['intensity']:.2f}</p>
                    <p>Valence: {emotional_state['valence']:.2f} | Arousal: {emotional_state['arousal']:.2f}</p>
                </div>
                """
                emotion_placeholder.markdown(emotion_html, unsafe_allow_html=True)
                
                # Display chart
                if show_emotion_data and len(st.session_state.emotion_history) > 1:
                    import pandas as pd
                    df = pd.DataFrame(st.session_state.emotion_history)
                    chart_placeholder.line_chart(df[['valence', 'arousal']])
            
            # Small delay
            time.sleep(0.03)
        
        cap.release()
else:
    # Show instructions when not running
    st.info("""
    ### 👋 Welcome to SynesthAI!
    
    **How it works:**
    1. Click "▶️ Start Experience" to begin
    2. Your webcam will activate and read your emotions
    3. Watch as art and music adapt to your feelings in real-time
    4. Try different expressions and see how the art changes!
    
    **Tips:**
    - Make sure you're in good lighting
    - Face the camera directly
    - Try smiling, looking sad, or showing different emotions
    - The art will evolve continuously with your emotional journey
    
    **This is YOUR emotional art experience. Every session is unique!**
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>SynesthAI</strong> - Where Emotion Becomes Art</p>
    <p>Created by MD Jawar Safi • AI for Creativity Final Project</p>
</div>
""", unsafe_allow_html=True)
