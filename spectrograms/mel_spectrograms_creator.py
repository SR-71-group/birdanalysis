import numpy as np
from scipy.signal import spectrogram
from pathlib import Path
from pydub import AudioSegment

def load_audio_with_padding(audio_file, target_length=60):
    """Load an audio file and pad/truncate it to the target length (seconds)."""
    try:
        audio = AudioSegment.from_file(audio_file)
        audio = audio.set_frame_rate(16000)  # Ensure consistent sampling rate

        # Convert to mono
        if audio.channels > 1:
            audio = audio.set_channels(1)

        # Calculate padding
        target_length_ms = target_length * 1000
        duration_ms = len(audio)

        if duration_ms < target_length_ms:
            silence = AudioSegment.silent(duration=target_length_ms - duration_ms)
            audio = audio + silence
        else:
            audio = audio[:target_length_ms]

        # Convert to NumPy array
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples /= np.iinfo(audio.array_type).max  # Normalize
        return samples, 16000  # Return audio data and sampling rate
    except Exception as e:
        print(f"Error loading audio file {audio_file}: {e}")
        return None, None

def generate_mel_filterbank(sr, n_fft, n_mels=128, fmax=8000):
    """Manually create a mel filterbank."""
    mel_min = 0
    mel_max = 2595 * np.log10(1 + fmax / 700)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    # Create the filterbank
    filterbank = np.zeros((n_mels, int(n_fft // 2 + 1)))
    for i in range(1, len(bin_points) - 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]
        filterbank[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        filterbank[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return filterbank

def compute_mel_spectrogram(audio, sr, n_fft=1024, n_mels=128, fmax=8000):
    """Compute a mel spectrogram without using librosa."""
    try:
        f, t, Sxx = spectrogram(audio, fs=sr, nperseg=n_fft, noverlap=n_fft // 2)
        mel_filter = generate_mel_filterbank(sr, n_fft, n_mels=n_mels, fmax=fmax)

        # Apply mel filter
        mel_spec = np.dot(mel_filter, Sxx)
        mel_spec_db = 10 * np.log10(np.maximum(mel_spec, 1e-10))  # Convert to dB scale
        return mel_spec_db
    except Exception as e:
        print(f"Error computing mel spectrogram: {e}")
        return None
