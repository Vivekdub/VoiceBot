from transformers import pipeline

# Load model (can be cached after first load)
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base")

def transcribe_audio(file_path):
    result = pipe(file_path)
    return result["text"]
