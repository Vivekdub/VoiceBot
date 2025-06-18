import vlc
import time

def play_audio_vlc(mp3_path):
    player = vlc.MediaPlayer(mp3_path)
    player.play()
    time.sleep(1)  # Give time for the player to start
    while player.is_playing():
        time.sleep(0.5)
play_audio_vlc("C:/Users/Shashankd/OneDrive/Desktop/New folder/homellc_stage1/murf_audio.mp3")  