from bark import generate_audio
import numpy as np
import scipy.io.wavfile
import tempfile

def generate_speech(text):
    audio_array = generate_audio(text)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    scipy.io.wavfile.write(temp_file.name, rate=22050, data=audio_array)
    return temp_file.name
