# spectrogram_padding.py
import numpy as np
from pathlib import Path

def pad_spectrogram(spectrogram, target_shape):
    """Pad the spectrogram to the target shape with zeros."""
    # Get current shape
    current_shape = spectrogram.shape

    # Calculate padding needed
    pad_width = [
        (0, max(0, target_shape[0] - current_shape[0])),  # Pad rows
        (0, max(0, target_shape[1] - current_shape[1]))   # Pad columns
    ]

    # Apply zero padding
    padded_spectrogram = np.pad(spectrogram, pad_width, mode='constant', constant_values=0)

    return padded_spectrogram

def process_and_pad_spectrograms(input_dir, output_dir, target_shape):
    """Load spectrograms, apply zero padding, and save them."""
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_name in input_dir.glob('*.npy'):
        # Load the spectrogram
        spectrogram = np.load(file_name)

        # Pad the spectrogram
        padded_spectrogram = pad_spectrogram(spectrogram, target_shape)

        # Save the padded spectrogram
        padded_filename = output_dir / file_name.name  # Keep the same name
        np.save(padded_filename, padded_spectrogram)
        print(f"Saved padded spectrogram as: {padded_filename}")
