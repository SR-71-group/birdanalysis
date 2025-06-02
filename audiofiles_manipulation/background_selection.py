import os
import random
from pathlib import Path
import wave
from collections import deque

def get_audio_duration(file_path):
    """Get the duration of a .wav file in seconds."""
    with wave.open(str(file_path), "rb") as wav_file:
        sr = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        return num_frames / sr, sr

def load_audio(file_path):
    """Load raw audio data from a .wav file."""
    with wave.open(str(file_path), "rb") as wav_file:
        return wav_file.readframes(wav_file.getnframes()), wav_file.getframerate(), wav_file.getsampwidth()

def save_audio(output_path, audio_data, sr, sample_width):
    """Save raw audio data to a .wav file."""
    with wave.open(str(output_path), "wb") as output_wav:
        output_wav.setnchannels(1)  # Assuming mono audio
        output_wav.setsampwidth(sample_width)  # Sample width (e.g., 16-bit PCM)
        output_wav.setframerate(sr)
        output_wav.writeframes(audio_data)

def preprocess_background_audio(input_dir, output_dir, target_samples=500, target_duration=60):
    """
    Preprocess background audio files to create unique 60-second audio clips.

    Args:
        input_dir (Path): Directory containing raw background noise files.
        output_dir (Path): Directory to save the processed 60-second audio clips.
        target_samples (int): Number of 60-second clips to create.
        target_duration (int): Duration of each clip in seconds.
    """

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all audio files in the input directory
    all_audio_files = list(input_dir.glob("*.wav"))
    if not all_audio_files:
        print(f"No .wav files found in the directory: {input_dir}")
        return  # Exit if no audio files are found

    audio_files_with_durations = []

    # Get the duration and sample rate for each audio file
    for file_path in all_audio_files:
        try:
            duration, sr = get_audio_duration(file_path)
            audio_files_with_durations.append((file_path, duration, sr))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not audio_files_with_durations:
        print(f"Unable to process any .wav files in {input_dir}.")
        return  # Exit if no files are successfully processed

    # Sort the files by duration (longest first)
    audio_files_with_durations.sort(key=lambda x: x[1], reverse=True)

    processed_count = 0
    short_audio_queue = deque()
    target_length_samples = target_duration * audio_files_with_durations[0][2]  # Target length in samples
    sample_width = 2  # Assuming 16-bit PCM

    # Process long files (those with duration >= target_duration)
    for file_path, duration, sr in audio_files_with_durations:
        if processed_count >= target_samples:
            break

        if duration >= target_duration:
            # Long file: Trim to 60 seconds
            audio_data, _, sample_width = load_audio(file_path)
            trimmed_audio = audio_data[:target_length_samples * sample_width]

            # Modify the filename to reflect the new `0-60s` duration
            base_name_parts = file_path.stem.split("___")
            duration_part_index = next((i for i, part in enumerate(base_name_parts) if "s" in part), -1)
            if duration_part_index != -1:
                base_name_parts[duration_part_index] = "0-60s"
            new_filename = "___".join(base_name_parts) + "___background.wav"
            output_path = output_dir / new_filename

            save_audio(output_path, trimmed_audio, sr, sample_width)
            print(f"Saved trimmed file: {output_path}")
            processed_count += 1

        else:
            # Short file: Add to the queue for concatenation
            short_audio_queue.append((file_path, duration, sr))

    # Concatenate short files to create 60-second clips if needed
    while processed_count < target_samples and short_audio_queue:
        concatenated_audio = bytearray()
        current_duration = 0
        sample_rate = short_audio_queue[0][2]  # Ensure sample rate consistency

        # Use the name of the first file in the queue as the base for the concatenated file
        first_file_path, _, _ = short_audio_queue[0]

        while short_audio_queue and current_duration < target_duration:
            file_path, duration, sr = short_audio_queue.popleft()
            if sr != sample_rate:
                print(f"Skipping file with mismatched sample rate: {file_path}")
                continue

            audio_data, _, _ = load_audio(file_path)
            concatenated_audio.extend(audio_data)
            current_duration += duration

        # Trim concatenated audio to exactly 60 seconds
        concatenated_audio = concatenated_audio[:target_length_samples * sample_width]

        # Modify the filename of the base file to reflect the new `0-60s` duration
        base_name_parts = first_file_path.stem.split("___")
        duration_part_index = next((i for i, part in enumerate(base_name_parts) if "s" in part), -1)
        if duration_part_index != -1:
            base_name_parts[duration_part_index] = "0-60s"
        new_filename = "___".join(base_name_parts) + ".wav"
        output_path = output_dir / new_filename

        save_audio(output_path, concatenated_audio, sample_rate, sample_width)
        print(f"Saved concatenated file: {output_path}")
        processed_count += 1

    print(f"Created {processed_count} 60-second background noise files in {output_dir}.")





if __name__ == "__main__":
    # Define input and output directories here
    base_dir = Path(__file__).resolve().parent.parent
    input_dir = base_dir / "data/audio/backgrounds"
    output_dir = base_dir / "data/preprocessed"
    # Call the function to preprocess background audio
    preprocess_background_audio(
        input_dir,
        output_dir,
        target_samples=500,  # Number of 60-second clips to create
        target_duration=60  # Duration of each clip in seconds
    )
