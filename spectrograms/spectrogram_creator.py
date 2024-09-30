# spectrogram_creator.py
import numpy as np
import librosa
from pathlib import Path

def create_spectrogram(audio_file, output_dir):
    """Generate and save a spectrogram from the given audio file as a numpy array."""
    # Load audio
    audio, sr = librosa.load(audio_file)

    # Create a spectrogram (short-time Fourier transform)
    spectrogram = np.abs(librosa.stft(audio))  # Magnitude of the STFT

    # Convert to decibels
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

    # Extract filename for saving
    file_name = Path(audio_file).stem  # Get file name without extension
    npy_filename = output_dir / f"{file_name}_spectrogram.npy"

    # Save the spectrogram as a numpy array
    np.save(npy_filename, spectrogram_db)
    print(f"Saved spectrogram as: {npy_filename}")
