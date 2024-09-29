# main.py
import os
from pathlib import Path
from audio_parser import parse_filename
from audio_processing import load_audio, cut_bird_call, create_background_noise
from file_saver import save_audio, generate_background_filename
from backgrounds_selection import filter_longest_files, delete_unselected_files
from spectrogram_creator import create_spectrogram  # Import the spectrogram creator
from spectrogram_padding import process_and_pad_spectrograms  # Import padding logic

# Get the root directory where the script is being executed
ROOT_DIR = Path(__file__).resolve().parent  # Get the current script directory

def process_audio_file(input_file):
    """Main process to handle an individual .wav file."""
    # Step 1: Parse the filename
    parsed_info = parse_filename(os.path.basename(input_file))

    # Step 2: Load the audio
    audio = load_audio(input_file)

    # Step 3: Extract the bird call timing from parsed_info
    start_time, end_time = map(float, parsed_info['time_range'].replace('s', '').split('-'))
    start_ms = int(start_time * 1000)  # Convert to milliseconds
    end_ms = int(end_time * 1000)

    # Step 4: Cut the bird call part
    bird_call_audio = cut_bird_call(audio, start_ms, end_ms)

    # Step 5: Create the background noise (without bird call)
    background_audio = create_background_noise(audio, len(bird_call_audio))

    # Step 6: Generate new filename for background audio
    new_filename = generate_background_filename(parsed_info)  # No time_range needed

    # Step 7: Save the background audio
    output_dir = ROOT_DIR / 'data' / 'backgrounds'  # Use Path object for output directory
    save_audio(background_audio, new_filename, output_dir)
    print(f"Processed and saved: {new_filename}")

def main():
    """Main entry point for processing all files in the audio directory."""
    # Dynamically construct the audio directory path
    audio_dir = ROOT_DIR / 'data' / 'audio'

    # Ensure the directory exists
    if not audio_dir.exists():
        raise FileNotFoundError(f"The audio directory {audio_dir} does not exist.")

    # Step 1: Process all audio files (cut bird calls and create background noise)
    for file_name in os.listdir(audio_dir):
        if file_name.endswith('.wav'):
            input_file = audio_dir / file_name  # Construct the full file path
            process_audio_file(input_file)

    # Step 2: Filter the top 1000 longest background audio files
    backgrounds_dir = ROOT_DIR / 'data' / 'backgrounds'
    longest_files = filter_longest_files(backgrounds_dir, top_n=1000)

    # Ensure the spectrograms output directory exists
    spectrogram_output_dir = ROOT_DIR / 'data' / 'spectrograms'
    spectrogram_output_dir.mkdir(parents=True, exist_ok=True)

    # Generate spectrograms only for the top 1000 longest background audio files
    for file_path in longest_files:
        create_spectrogram(file_path, spectrogram_output_dir)

    # Step 3: Apply zero padding to the spectrograms
    target_shape = (128, 256)  # Set the desired shape for the spectrograms

    # Create a new directory for padded spectrograms
    padded_spectrograms_dir = ROOT_DIR / 'data' / 'padded_spectrograms'
    padded_spectrograms_dir.mkdir(parents=True, exist_ok=True)

    # Process and pad spectrograms, saving them in the new folder
    process_and_pad_spectrograms(spectrogram_output_dir, padded_spectrograms_dir, target_shape)

if __name__ == "__main__":
    main()
