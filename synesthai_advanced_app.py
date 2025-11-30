"""
SynesthAI ADVANCED - Ultimate Application
The most powerful emotion-to-art AI experience
"""

import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Import our advanced modules
from advanced_emotion_detector import AdvancedEmotionDetector
from advanced_visual_art import AdvancedVisualArtGenerator
from advanced_music import AdvancedMusicGenerator

# Page configuration
st.set_page_config(
    page_title="SynesthAI Advanced",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 4em;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 20px #667eea; }
        to { text-shadow: 0 0 30px #764ba2, 0 0 40px #667eea; }
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.5em;
        color: #666;
        margin-bottom: 20px;
        font-weight: 300;
    }
    
    .emotion-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        margin: 10px 0;
    }
    
    .stat-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    .metric-value {
        font-size: 2.5em;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.9em;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .status-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9em;
    }
    
    .status-active {
        background: #4caf50;
        color: white;
    }
    
    .status-paused {
        background: #ff9800;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">✨ SynesthAI Advanced</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Where Your Emotions Become Living Art</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Advanced Settings")
    
    # Mode selection
    st.subheader("Experience Mode")
    session_mode = st.selectbox(
        "Choose Mode",
        ["🎨 Full Experience", "🧪 Emotion Lab", "📊 Analytics Mode", "🎬 Recording Mode"]
    )
    
    st.markdown("---")
    
    # Feature toggles
    st.subheader("🎛️ Features")
    enable_music = st.checkbox("🎵 Advanced Music", value=True)
    enable_effects = st.checkbox("✨ Special Effects", value=True)
    show_analytics = st.checkbox("📊 Live Analytics", value=True)
    show_history = st.checkbox("📈 Emotion History", value=True)
    
    st.markdown("---")
    
    # Visual settings
    st.subheader("🎨 Visual Settings")
    particle_density = st.slider("Particle Density", 100, 1000, 500)
    art_complexity = st.select_slider(
        "Art Complexity",
        options=["Minimal", "Balanced", "Rich", "Maximum"],
        value="Rich"
    )
    
    st.markdown("---")
    
    # Info
    st.subheader("ℹ️ About")
    st.info("""
    **SynesthAI Advanced** uses cutting-edge AI to transform your emotions into:
    
    - 🎨 Dynamic generative art
    - 🎵 Rich procedural music
    - 📊 Real-time analytics
    - 💫 Multi-sensory experience
    
    **Tech Stack:**
    - DeepFace emotion AI
    - YOLOv8 pose detection
    - Advanced particle systems
    - Procedural music synthesis
    """)
    
    st.caption("MD Jawar Safi • AI for Creativity")

# Initialize session state
if 'advanced_emotion_detector' not in st.session_state:
    with st.spinner("🚀 Loading Advanced AI Models..."):
        st.session_state.advanced_emotion_detector = AdvancedEmotionDetector()
        st.session_state.advanced_art_generator = AdvancedVisualArtGenerator(width=1200, height=800)
        st.session_state.advanced_music_generator = AdvancedMusicGenerator()
    st.success("✅ Advanced AI Models Loaded!")

if 'running' not in st.session_state:
    st.session_state.running = False

if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []

if 'session_data' not in st.session_state:
    st.session_state.session_data = {
        'start_time': None,
        'total_frames': 0,
        'emotions_detected': {}
    }

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["🎨 Live Experience", "📊 Analytics Dashboard", "🎬 Session Replay"])

with tab1:
    # Top status bar
    col_status1, col_status2, col_status3 = st.columns(3)
    
    with col_status1:
        status = "Active" if st.session_state.running else "Paused"
        status_class = "status-active" if st.session_state.running else "status-paused"
        st.markdown(f'<span class="status-badge {status_class}">● {status}</span>', unsafe_allow_html=True)
    
    with col_status2:
        frames = st.session_state.session_data['total_frames']
        st.markdown(f"**Frames Processed:** {frames}")
    
    with col_status3:
        if st.session_state.session_data['start_time']:
            elapsed = (datetime.now() - st.session_state.session_data['start_time']).total_seconds()
            st.markdown(f"**Session Time:** {int(elapsed)}s")
    
    st.markdown("---")
    
    # Main display area
    col_video, col_art = st.columns([1, 1.5])
    
    with col_video:
        st.subheader("📹 Live Video Feed")
        video_placeholder = st.empty()
    
    with col_art:
        st.subheader("🎨 Emotional Art Canvas")
        art_placeholder = st.empty()
    
    # Emotion display
    st.markdown("---")
    emotion_col1, emotion_col2 = st.columns([1, 2])
    
    with emotion_col1:
        st.subheader("😊 Current Emotion")
        emotion_display = st.empty()
    
    with emotion_col2:
        if show_analytics:
            st.subheader("📊 Emotional Metrics")
            metrics_placeholder = st.empty()

# Control buttons
st.markdown("---")
button_col1, button_col2, button_col3, button_col4 = st.columns(4)

with button_col1:
    if st.button("▶️ Start Experience", use_container_width=True, type="primary"):
        st.session_state.running = True
        st.session_state.session_data['start_time'] = datetime.now()
        if enable_music:
            st.session_state.advanced_music_generator.start()

with button_col2:
    if st.button("⏸️ Pause", use_container_width=True):
        st.session_state.running = False
        st.session_state.advanced_music_generator.stop()

with button_col3:
    if st.button("🔄 Reset Session", use_container_width=True):
        st.session_state.advanced_art_generator = AdvancedVisualArtGenerator(width=1200, height=800)
        st.session_state.emotion_history = []
        st.session_state.session_data = {
            'start_time': None,
            'total_frames': 0,
            'emotions_detected': {}
        }
        st.session_state.running = False
        st.session_state.advanced_music_generator.stop()
        st.rerun()

with button_col4:
    if st.button("💾 Save Session", use_container_width=True):
        st.info("Session recording will be saved!")

# Main processing loop
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Could not open webcam")
    else:
        frame_count = 0
        
        while st.session_state.running:
            ret, frame = cap.read()
            
            if not ret:
                st.error("❌ Could not read frame")
                break
            
            frame_count += 1
            st.session_state.session_data['total_frames'] += 1
            
            # Process every 2nd frame
            if frame_count % 2 == 0:
                # Detect emotions with advanced detector
                emotional_state = st.session_state.advanced_emotion_detector.detect_emotions(frame)
                
                # Track emotions
                emotion = emotional_state['primary_emotion']
                st.session_state.session_data['emotions_detected'][emotion] = \
                    st.session_state.session_data['emotions_detected'].get(emotion, 0) + 1
                
                # Update music
                if enable_music:
                    st.session_state.advanced_music_generator.update_emotion(emotional_state)
                
                # Generate advanced art
                art_frame = st.session_state.advanced_art_generator.generate_frame(emotional_state)
                
                # Save to history
                st.session_state.emotion_history.append({
                    'time': datetime.now(),
                    'emotion': emotion,
                    'valence': emotional_state['valence'],
                    'arousal': emotional_state['arousal'],
                    'intensity': emotional_state['intensity'],
                    'confidence': emotional_state['confidence']
                })
                
                # Keep last 200 points
                if len(st.session_state.emotion_history) > 200:
                    st.session_state.emotion_history.pop(0)
                
                # Display video
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Add emotion overlay to video
                description = st.session_state.advanced_emotion_detector.get_emotion_description(emotional_state)
                cv2.putText(frame_rgb, description, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Display art
                art_rgb = cv2.cvtColor(art_frame, cv2.COLOR_BGR2RGB)
                art_placeholder.image(art_rgb, channels="RGB", use_column_width=True)
                
                # Display emotion card
                emotion_html = f"""
                <div class="emotion-card">
                    <h2 style="margin: 0; font-size: 2.5em;">{emotion.upper()}</h2>
                    <p style="font-size: 1.2em; margin: 10px 0;">
                        {description}
                    </p>
                    <p style="opacity: 0.9;">
                        Confidence: {emotional_state['confidence']:.1%} | 
                        Posture: {emotional_state['body_posture']}
                    </p>
                </div>
                """
                emotion_display.markdown(emotion_html, unsafe_allow_html=True)
                
                # Display metrics
                if show_analytics:
                    metric_cols = metrics_placeholder.columns(4)
                    
                    with metric_cols[0]:
                        st.metric("Valence", f"{emotional_state['valence']:.2f}", 
                                 delta=emotional_state.get('emotional_trajectory', 'stable'))
                    
                    with metric_cols[1]:
                        st.metric("Arousal", f"{emotional_state['arousal']:.2f}")
                    
                    with metric_cols[2]:
                        st.metric("Intensity", f"{emotional_state['intensity']:.1%}")
                    
                    with metric_cols[3]:
                        st.metric("Confidence", f"{emotional_state['confidence']:.1%}")
            
            time.sleep(0.03)
        
        cap.release()
else:
    # Show welcome screen
    st.info("""
    ### 🌟 Welcome to SynesthAI Advanced!
    
    **This is the most powerful emotion-to-art AI ever created.**
    
    **Features:**
    - 🧠 Advanced emotion detection with DeepFace
    - 🎨 Multi-layered generative art
    - 🎵 Rich procedural music with harmonies
    - 📊 Real-time analytics dashboard
    - 💾 Session recording and replay
    
    **How to use:**
    1. Click "▶️ Start Experience"
    2. Allow camera access
    3. Express different emotions
    4. Watch your feelings transform into art!
    
    **Adjust settings in the sidebar for a customized experience.**
    """)

# Analytics tab
with tab2:
    if len(st.session_state.emotion_history) > 0:
        st.subheader("📈 Emotional Journey Visualization")
        
        # Create DataFrame
        df = pd.DataFrame(st.session_state.emotion_history)
        
        # Plot valence and arousal over time
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df['valence'], name='Valence', line=dict(color='#667eea', width=3)))
        fig.add_trace(go.Scatter(y=df['arousal'], name='Arousal', line=dict(color='#764ba2', width=3)))
        fig.update_layout(height=400, title="Emotional Trajectory", 
                         xaxis_title="Time", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)
        
        # Emotion distribution
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            emotion_counts = df['emotion'].value_counts()
            fig2 = px.pie(values=emotion_counts.values, names=emotion_counts.index,
                         title="Emotion Distribution")
            st.plotly_chart(fig2, use_container_width=True)
        
        with col_chart2:
            fig3 = px.scatter(df, x='valence', y='arousal', color='emotion',
                            title="Valence-Arousal Space", size='intensity')
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("📊 Start an experience to see analytics!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <h3 style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        SynesthAI Advanced
    </h3>
    <p><strong>The Future of Emotional AI Art</strong></p>
    <p>Created by MD Jawar Safi • AI for Creativity Final Project</p>
    <p style='font-size: 0.9em;'>Powered by DeepFace, YOLOv8, Advanced Algorithms</p>
</div>
""", unsafe_allow_html=True)
