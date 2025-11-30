# 🎨 SynesthAI - The AI That Turns Emotions Into Living Art

<div align="center">

![SynesthAI Banner](https://img.shields.io/badge/AI-Emotion%20To%20Art-blueviolet?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

**A revolutionary AI system that reads your emotions in real-time and creates a unique multi-sensory artistic experience**

[Demo Video](#) • [Live Demo](#) • [Documentation](#) • [Report Issue](https://github.com/786jawar/SynesthAI/issues)

</div>

---

## 🌟 What is SynesthAI?

SynesthAI is the world's first AI that combines real-time emotion detection with multi-modal art generation. It reads your facial expressions and body language, then creates:

- 🎨 **Dynamic Visual Art** - Generative particle systems that evolve with your feelings
- 🎵 **Procedural Music** - Rich harmonies that adapt to your emotional state  
- 📊 **Real-time Analytics** - Live visualization of your emotional journey
- 💫 **Unique Experiences** - Every session creates unrepeatable art

---

## ✨ Features

### 🧠 Advanced Emotion Detection
- **DeepFace AI** for facial emotion recognition
- **YOLOv8-Pose** for body language analysis
- **7 emotion types** detected (happy, sad, angry, fear, surprise, disgust, neutral)
- **Temporal smoothing** for stable predictions
- **Confidence scoring** for reliability

### 🎨 Multi-Layer Visual Art
- **500 particle physics system** with flow fields
- **Dynamic gradients** based on emotional valence
- **Motion trails** and glow effects
- **Emotion-specific effects** (sparkles for joy, rain for sadness)
- **3 rendering layers** for depth

### 🎵 Rich Procedural Music
- **Chord progressions** that match your mood
- **5 harmonic overtones** for rich sound
- **Multiple musical scales** (major, minor, pentatonic)
- **Adaptive tempo** based on arousal level
- **ADSR envelopes** for professional sound

### 💻 Beautiful Interface
- **Streamlit-powered** web interface
- **Real-time charts** with Plotly
- **Session recording** and replay
- **Customizable settings**
- **Analytics dashboard**

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
Webcam
Windows/Mac/Linux
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/786jawar/SynesthAI.git
cd SynesthAI
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run synesthai_advanced_app.py
```

4. **Open your browser** (should open automatically at `localhost:8501`)

5. **Click "Start Experience"** and express your emotions!

---

## 📖 How It Works

```
┌─────────────────────────────────────────┐
│         Webcam Input                    │
└────────────────┬────────────────────────┘
                 │
     ┌───────────┴──────────┐
     │  Emotion Detector     │
     │  - DeepFace AI        │
     │  - YOLO Pose          │
     │  - Valence/Arousal    │
     └───────────┬───────────┘
                 │
     ┌───────────┴───────────┐
     │   Emotion State       │
     │  {emotion, valence,    │
     │   arousal, intensity}  │
     └───────┬───────┬────────┘
             │       │
     ┌───────┘       └───────┐
     │                       │
┌────▼─────────┐    ┌────────▼────────┐
│  Art Gen     │    │  Music Gen      │
│  - Particles │    │  - Harmonies    │
│  - Flow      │    │  - Chords       │
│  - Effects   │    │  - Melody       │
└────┬─────────┘    └────────┬────────┘
     │                       │
     └───────────┬───────────┘
                 │
     ┌───────────▼───────────┐
     │   Streamlit UI        │
     │  - Video Display      │
     │  - Art Canvas         │
     │  - Analytics          │
     └───────────────────────┘
```

---

## 🎯 Use Cases

### 🧘 Mental Health Therapy
- PTSD treatment through safe emotional expression
- Visual record of healing journey
- Real-time adaptation to emotional breakthroughs
- **Market:** $10M (1,000 therapy clinics)

### 🏛️ Museum Installations
- "You Are The Art" exhibitions
- 10-minute emotional pod experiences
- NFT certificates of unique artwork
- **Market:** $5M (100 museums globally)

### 🎤 Live Music Performances
- AI generates visuals during concerts
- Unique art for each performance
- Fans collect NFTs from shows
- **Market:** Unlimited potential

### 🧘 Meditation Apps
- Calming art adapts to stress levels
- Visual biofeedback
- **Market:** $1.5M monthly (100K users)

---

## 🛠️ Technology Stack

- **Python 3.8+** - Core language
- **Streamlit** - Web framework
- **DeepFace** - Facial emotion recognition
- **Ultralytics YOLOv8** - Pose detection
- **OpenCV** - Computer vision
- **PyGame** - Audio synthesis
- **NumPy** - Numerical computing
- **Plotly** - Interactive charts
- **Pandas** - Data analysis

---

## 📊 Project Structure

```
SynesthAI/
├── advanced_emotion_detector.py    # Emotion detection engine
├── advanced_visual_art.py          # Art generation system
├── advanced_music.py               # Music composition
├── synesthai_advanced_app.py       # Main Streamlit app
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── LICENSE                         # MIT License
└── docs/                          # Documentation
    ├── architecture.md
    ├── api_reference.md
    └── user_guide.md
```

---

## 🎨 Emotion-to-Art Mapping

| Emotion   | Color       | Movement      | Music       |
|-----------|-------------|---------------|-------------|
| Happy     | Yellow/Orange | Rising, energetic | Major scale, high |
| Sad       | Blue        | Falling, slow | Minor scale, low |
| Angry     | Red         | Chaotic, jagged | Tense, dissonant |
| Fear      | Purple      | Scattered     | Unsettled |
| Surprise  | Cyan        | Burst         | Varied |
| Neutral   | Gray        | Balanced      | Calm |

---

## 📈 Performance

- **30 FPS** emotion detection
- **500 particles** real-time rendering
- **< 100ms latency** from emotion to art
- **7 emotions** recognized
- **9 body postures** analyzed

---

## 🔮 Future Enhancements

- [ ] Stable Diffusion integration for photorealistic art
- [ ] GPT-4 poetic narration
- [ ] Haptic feedback via phone/smartwatch
- [ ] NFT minting of sessions
- [ ] Multi-user collaborative experiences
- [ ] VR/AR immersive mode
- [ ] MusicGen AI for advanced composition

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**MD Jawar Safi**
- 🎓 AI for Creativity - Final Project
- 📧 Email: 2315024@students.ucreative.ac.uk
- 💻 GitHub: [@786jawar](https://github.com/786jawar)
- 🔗 Portfolio: [github.com/786jawar/SynesthAI](https://github.com/786jawar/SynesthAI)

---

## 🙏 Acknowledgments

- **DeepFace** - Facial emotion recognition
- **Ultralytics** - YOLO pose detection
- **Streamlit** - Web framework
- **OpenCV** - Computer vision tools
- **My professors** - For guidance and support

---

## 📞 Contact & Support

- 🐛 **Bug Reports:** [Open an issue](https://github.com/786jawar/SynesthAI/issues)
- 💡 **Feature Requests:** [Open an issue](https://github.com/786jawar/SynesthAI/issues)
- 📧 **Email:** 2315024@students.ucreative.ac.uk

---

## ⭐ Show Your Support

If you find SynesthAI interesting, please consider:
- ⭐ **Starring** this repository
- 🍴 **Forking** to build upon it
- 📢 **Sharing** with others
- 💬 **Contributing** improvements

---

<div align="center">

**SynesthAI - Where Emotion Becomes Art**

Made with ❤️ and 🤖 by MD Jawar Safi

[⬆ Back to Top](#-synesthai---the-ai-that-turns-emotions-into-living-art)

</div>
