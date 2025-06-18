import streamlit as st
from st_audiorec import st_audiorec
import tempfile
import requests
from murf import Murf
import pygame
import time
import os
import soundfile as sf
import numpy as np
from transformers import pipeline
import vlc
from dotenv import load_dotenv
load_dotenv()
# -------------------- API Keys --------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
MURF_API_KEY = os.getenv("MURF_API_KEY")
if not OPENROUTER_API_KEY or not HF_API_KEY or not MURF_API_KEY:
    st.error("❌ One or more API keys are missing. Check your environment variables in Render.")
    st.stop()
# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="🎙️ Voice GPT Assistant", layout="centered")
st.markdown("<h1 style='text-align: center;'>🎙️ Talk to GPT-4o</h1>", unsafe_allow_html=True)

# -------------------- TTS Function (Murf) --------------------
def generate_speech(text, voice_id="en-US-terrell", output_file="murf_audio.mp3"):
    try:
        client = Murf(api_key=MURF_API_KEY)
        res = client.text_to_speech.generate(text=text, voice_id=voice_id)
        audio_url = res.audio_file

        r = requests.get(audio_url)
        if r.status_code == 200:
            with open(output_file, "wb") as f:
                f.write(r.content)
            return output_file
        else:
            st.error("❌ Failed to download audio.")
            return None
    except Exception as e:
        st.error(f"❌ TTS error: {e}")
        return None

# -------------------- STT Pipeline --------------------
@st.cache_resource
def load_stt_pipeline():
    return pipeline("automatic-speech-recognition", model="openai/whisper-small")

# -------------------- Play MP3 Audio via Pygame --------------------
def play_audio_mp3(mp3_path):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(mp3_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        st.error(f"❌ Audio playback failed: {e}")

def play_audio_vlc(mp3_path):
    player = vlc.MediaPlayer(mp3_path)
    player.play()
    time.sleep(1)  # Give time for the player to start
    while player.is_playing():
        time.sleep(0.5)
# -------------------- UI --------------------
st.markdown("### Click the mic and speak")
audio_bytes = st_audiorec()

if audio_bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        temp_audio_file.write(audio_bytes)
        temp_audio_path = temp_audio_file.name

    # Convert to mono if needed
    audio_data, sample_rate = sf.read(temp_audio_path)
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
        sf.write(temp_audio_path, audio_data, sample_rate)

    # -------------------- STT --------------------
    stt_pipeline = load_stt_pipeline()
    result = stt_pipeline(temp_audio_path)
    transcript = result['text']
    st.markdown(f"**🗣️ You said:** `{transcript}`")

    # -------------------- GPT-4o Call --------------------
    st.markdown("**🤖 Thinking...**")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful voice assistant."},
            {"role": "user", "content": transcript}
        ],
        "max_tokens": 1000
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)

    if response.ok:
        reply = response.json()["choices"][0]["message"]["content"]
        st.markdown(f"**🤖 GPT-4o:** {reply}")

        # -------------------- TTS --------------------
        tts_file = generate_speech(reply)
        if tts_file and os.path.exists(tts_file):
            play_audio_vlc(tts_file)
        else:
            st.error("TTS failed or the file is corrupted.")
    else:
        st.error("❌ GPT-4o API call failed")
        st.text(response.text)
