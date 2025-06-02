import numpy as np
import pandas as pd
from pathlib import Path
from spectrograms.mel_spectrograms_creator import load_audio_with_padding, compute_mel_spectrogram

# ❗ PRECOMPUTING FEATURE VECTORS COULD REDUCE COMPUTATIONAL COSTS BUT NEEDS A LOT OF DISK SPACE WHICH CAN BE LIMITED ESPECIALLY WHEN USED GOOGLE COLAB OR SO ❗

def process_main_audio(csv_path, audio_dir, output_dir, max_duration=60):
    """Process audio files for base calls and save their mel-spectrograms."""
    try:
        # Load metadata CSV
        df = pd.read_csv(csv_path)

        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for _, row in df.iterrows():
            audio_file = Path(audio_dir) / row["filename"]
            if not audio_file.exists():
                print(f"File not found: {audio_file}")
                continue

            # Load audio and compute mel-spectrogram
            audio_data, sample_rate = load_audio_with_padding(audio_file, max_duration)
            if audio_data is not None:
                mel_spectrogram = compute_mel_spectrogram(audio_data, sample_rate)
                if mel_spectrogram is not None:
                    # Save mel-spectrogram as .npy
                    file_name = audio_file.stem
                    npy_filename = output_dir / f"{file_name}.npy"
                    np.save(npy_filename, mel_spectrogram)
                    print(f"Saved mel-spectrogram: {npy_filename}")

        # Save updated CSV with filenames
        updated_csv = output_dir / "updated_metadata.csv"
        df.to_csv(updated_csv, index=False)
        print(f"Saved updated metadata CSV: {updated_csv}")

    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "metadata/sliding_window_annotations.csv"
    audio_dir = base_dir / "preprocessed"
    output_dir = base_dir / "mel-spectrograms"

    process_main_audio(csv_path, audio_dir, output_dir)
