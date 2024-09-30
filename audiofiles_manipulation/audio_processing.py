
from pydub import AudioSegment

MAX_DURATION_MS = 2 * 60 * 1000 + 30 * 1000  # 2 minutes 30 seconds in milliseconds

def load_audio(file_path):
    """Load an audio file using pydub."""
    return AudioSegment.from_wav(file_path)

def cut_bird_call(audio, start_ms, end_ms):
    """Cut the portion of the audio that contains the bird call."""
    return audio[start_ms:end_ms]

def create_background_noise(audio, bird_call_duration):
    """
    Create a background noise file that lasts as long as MAX_DURATION_MS.
    If the file is too short, just return whatever background noise is left.
    """
    if len(audio) >= bird_call_duration:
        return audio[:MAX_DURATION_MS]
    return audio

# Example usage:
# audio = load_audio('data/audio/2459995.723424_Tautenburg___1652-6323kHz___10-20.3s___s..wav')
# bird_call = cut_bird_call(audio, start_ms=10000, end_ms=20300)  # 10 to 20.3 seconds
# background = create_background_noise(audio, len(bird_call))
