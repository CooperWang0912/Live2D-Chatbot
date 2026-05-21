import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

model = WhisperModel(
    "Systran/faster-distil-whisper-small.en",  # pre-converted for faster-whisper
    device="cpu",
    compute_type="int8"
)

def record_once(sample_rate=16000, silence_threshold=500, silence_duration=0.5):
    audio_chunks = []
    silent_samples = 0
    chunk_size = 1024

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:
        print("Listening...")

        # Wait for speech to begin
        while True:
            chunk, _ = stream.read(chunk_size)
            if np.max(np.abs(chunk)) > silence_threshold:
                audio_chunks.append(chunk.copy())
                break

        # Record until silence
        while True:
            chunk, _ = stream.read(chunk_size)
            audio_chunks.append(chunk.copy())
            if np.max(np.abs(chunk)) < silence_threshold:
                silent_samples += chunk_size
                if silent_samples >= silence_duration * sample_rate:
                    break
            else:
                silent_samples = 0

    audio = np.concatenate(audio_chunks).flatten().astype(np.float32) / 32768.0
    segments, _ = model.transcribe(audio)
    return " ".join(s.text for s in segments).strip()

if __name__ == "__main__":
    print(record_once())