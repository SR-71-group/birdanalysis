# bird_call_extractor.py
import os
from pathlib import Path
from audio_parser import parse_filename
from audio_processing import load_audio, cut_bird_call
from file_saver import save_audio
from spectrogram_creator import create_spectrogram
from spectrogram_padding import process_and_pad_spectrograms

# Get the root directory where the script is being executed
ROOT_DIR = Path(__file__).resolve().parent  # Get the current script directory

def extract_bird_calls():
    """Extract bird calls from audio files and generate spectrograms."""
    audio_dir = ROOT_DIR / 'data' / 'audio'
    preprocessed_dir = ROOT_DIR / 'data' / 'preprocessed'
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    for file_name in os.listdir(audio_dir):
        if file_name.endswith('.wav'):
            input_file = audio_dir / file_name
            parsed_info = parse_filename(file_name)

            # Load the audio
            audio = load_audio(input_file)

            # Extract bird call timing
            start_time, end_time = map(float, parsed_info['time_range'].replace('s', '').split('-'))
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)

            # Cut the bird call part
            bird_call_audio = cut_bird_call(audio, start_ms, end_ms)

            # Save the bird call audio
            save_audio(bird_call_audio, file_name, preprocessed_dir)
            print(f"Saved bird call audio: {file_name} to {preprocessed_dir}")

    # Create spectrograms for preprocessed bird call audio
    spectrogram_output_dir = ROOT_DIR / 'data' / 'padded_spectrograms'
    spectrogram_output_dir.mkdir(parents=True, exist_ok=True)

    for file_name in os.listdir(preprocessed_dir):
        if file_name.endswith('.wav'):
            audio_file = preprocessed_dir / file_name
            create_spectrogram(audio_file, spectrogram_output_dir)

    target_shape = (128, 256)
    process_and_pad_spectrograms(spectrogram_output_dir, spectrogram_output_dir, target_shape)

def main():
    """Main entry point for the bird call extraction script."""
    extract_bird_calls()

if __name__ == "__main__":
    main()
