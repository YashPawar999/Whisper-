import sounddevice as sd
import numpy as np
import queue
from faster_whisper import WhisperModel

# =====================
# CONFIG
# =====================
SAMPLE_RATE = 16000
CHUNK_DURATION = 1.2        
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

SILENCE_THRESHOLD = 0.015   
MIN_ENERGY = 0.0005         


model = WhisperModel(
    "base.en",
    device="cpu",
    compute_type="int8"
)

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

print("🎙️ Listening... Speak English")
print("Press Ctrl+C to stop")

try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=CHUNK_SIZE,
        callback=audio_callback
    ):
        while True:
            audio_chunk = audio_queue.get().flatten()

            # ===== Noise Filtering =====

            
            if np.max(np.abs(audio_chunk)) < SILENCE_THRESHOLD:
                continue

            
            energy = np.mean(audio_chunk ** 2)
            if energy < MIN_ENERGY:
                continue

            audio_chunk = audio_chunk.astype(np.float32)

            segments, info = model.transcribe(
                audio_chunk,
                language="en",
                beam_size=2,         
                temperature=0.0,     
                vad_filter=True
            )

            for segment in segments:
                text = segment.text.strip()
                if len(text) > 2:
                    print("📝", text)

except KeyboardInterrupt:
    print("\nStopped by user")

print("Program exited cleanly ✅")
