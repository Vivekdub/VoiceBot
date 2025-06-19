import streamlit as st
from st_audiorec import st_audiorec
import requests
import os
import soundfile as sf
import numpy as np
import time
import vlc
from murf import Murf
from dotenv import load_dotenv

# -------------------- Load Environment Variables --------------------
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
MURF_API_KEY = os.getenv("MURF_API_KEY")

if not OPENROUTER_API_KEY or not HF_API_KEY or not MURF_API_KEY:
    st.error("❌ One or more API keys are missing. Check your environment variables.")
    st.stop()

# -------------------- Streamlit Page Config --------------------
st.set_page_config(page_title="🎙️ Voice GPT Assistant", layout="centered")
st.markdown("<h1 style='text-align: center;'>🎙️ Talk to GPT-4o</h1>", unsafe_allow_html=True)

# -------------------- Whisper STT (REST API via requests) --------------------
WHISPER_API_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3"

def query_whisper_rest(flac_path):
    with open(flac_path, "rb") as f:
        data = f.read()

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "audio/flac"
    }

    response = requests.post(WHISPER_API_URL, headers=headers, data=data)

    try:
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"❌ Whisper REST API failed: {e}")
        st.text(response.text)
        return None

# -------------------- TTS via Murf --------------------
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
            st.error("❌ Failed to download audio from Murf.")
            return None
    except Exception as e:
        st.error(f"❌ Murf TTS error: {e}")
        return None

# -------------------- Audio Playback via VLC --------------------
def play_audio_vlc(mp3_path):
    try:
        player = vlc.MediaPlayer(mp3_path)
        player.play()
        time.sleep(1)
        while player.is_playing():
            time.sleep(0.5)
    except Exception as e:
        st.error(f"❌ Audio playback failed: {e}")

# -------------------- Streamlit UI --------------------
st.markdown("### Click the Start recording to record and Stop to submit the audio")
audio_bytes = st_audiorec()

if audio_bytes:
    with st.spinner("🎙️ Processing your voice..."):

        # Step 1: Save WAV
        wav_path = os.path.join(os.getcwd(), "recorded_audio.wav")
        with open(wav_path, "wb") as f:
            f.write(audio_bytes)
        #st.text(f"🔊 Saved: {wav_path}")

        # Step 2: Convert to FLAC
        try:
            audio_data, sample_rate = sf.read(wav_path)
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)  # convert to mono
            flac_path = os.path.join(os.getcwd(), "converted.flac")
            sf.write(flac_path, audio_data, sample_rate, format="FLAC")
            #st.text(f"🎧 Saved FLAC: {flac_path}")
        except Exception as e:
            st.error(f"❌ Error converting to FLAC: {e}")
            st.stop()

        # Step 3: Transcribe using Whisper REST API
        result = query_whisper_rest(flac_path)
        if result and "text" in result:
            transcript = result["text"]
            st.markdown(f"**🗣️ You said:** `{transcript}`")
        else:
            st.error("❌ Failed to transcribe audio.")
            st.stop()

    # Step 4: GPT-4o Response
    with st.spinner("🤖 GPT-4o is thinking..."):
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
        else:
            st.error("❌ GPT-4o API call failed")
            st.text(response.text)
            st.stop()

    # Step 5: TTS + Playback
    tts_file = generate_speech(reply)
    if tts_file and os.path.exists(tts_file):
        with open(tts_file, "rb") as f:
            st.audio(f.read(), format="audio/mp3")
    else:
        st.error("❌ TTS failed or file not found.")
