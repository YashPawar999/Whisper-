# Whisper-
Whisper model for Speech to Text Recognition 
# 🎙️ Real-Time English Speech-to-Text (CPU) using Faster-Whisper

## 📌 Overview

This project implements a **real-time English speech-to-text system** using a microphone and a pretrained Whisper model.

It captures live audio from the system microphone, filters noise, and transcribes spoken English into text using **Faster-Whisper** optimized for CPU execution.

This system can be used in:

* Voice assistants
* Accessibility tools
* Meeting transcription systems
* Voice-controlled applications
* Game voice command systems
* AI chatbot input pipelines

---

## ⚙️ How It Works

### 1. Audio Capture

The system continuously records audio from the microphone using `sounddevice`.

Audio is captured in small chunks (~1.2 seconds).

```
Microphone → Audio Chunk → Processing Pipeline
```

---

### 2. Noise Filtering

Before sending audio to the model, the system removes low-energy signals:

* Peak amplitude threshold removes silence
* Energy threshold removes background noise

This reduces unnecessary transcription calls and improves accuracy.

---

### 3. Speech Recognition Model

The project uses:

**Faster-Whisper (`base.en`)**

* Transformer-based ASR model
* Pretrained on large multilingual speech datasets
* Optimized for CPU inference using INT8 quantization

The model converts speech directly into English text without needing phonemes, HMMs, or manual training.

---

### 4. Transcription Pipeline

```
Mic Input
   ↓
Audio Chunking
   ↓
Noise Filtering
   ↓
Whisper Encoder–Decoder
   ↓
Text Output
```

Each valid chunk is transcribed and printed to the terminal.

---

## 🖥️ Requirements

### Python

* Python 3.9 – 3.11 recommended

### Install dependencies

```bash
pip install faster-whisper sounddevice numpy
```

If needed:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## ▶️ How To Run

1. Clone repository

```bash
git clone <repo_url>
cd <repo_name>
```

2. Run the script

```bash
python realtime_whisper.py
```

3. Speak into the microphone
4. Transcription appears in terminal
5. Stop with:

```
Ctrl + C
```

---

## 🔌 How To Integrate Into Another Project

### Option 1 — Import as a module

Refactor the transcription loop into a function:

```python
def transcribe_audio_chunk(audio_chunk):
    segments, _ = model.transcribe(
        audio_chunk,
        language="en",
        beam_size=2,
        temperature=0.0,
        vad_filter=True
    )
    return " ".join([s.text.strip() for s in segments])
```

You can then call this function from:

* a GUI app
* a web API
* a game engine bridge
* a chatbot backend
* a real-time subtitle system

---

### Option 2 — Use as a background worker

The script can be wrapped inside:

* Flask / FastAPI server
* WebSocket streaming service
* Desktop assistant
* Game voice input thread

Example integration flow:

```
Frontend Mic → Backend Python Service → Whisper → Text → Application Logic
```

---

## 📂 Project Structure

```
project/
│── realtime_whisper.py
│── README.md
│── requirements.txt (optional)
```

---

## 🚀 Possible Improvements

* Continuous streaming transcription (rolling buffer)
* Sentence segmentation
* Punctuation restoration
* Voice activity detection tuning
* GPU acceleration support
* REST API wrapper
* GUI interface
* Multilingual support

---

## 🧠 Model Notes

| Feature | Value                          |
| ------- | ------------------------------ |
| Model   | Whisper base.en                |
| Runtime | CPU (INT8)                     |
| Latency | ~1–3 sec depending on hardware |
| Input   | Microphone audio               |
| Output  | English text                   |

---

## 📜 License

Use according to Whisper and Faster-Whisper licenses.

---

## 👤 Author

Developed as a real-time ASR prototype for integration into AI and voice-enabled systems.
