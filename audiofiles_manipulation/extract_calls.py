import os
import re
import wave
from pathlib import Path

def extract_duration_and_save(input_dir, output_dir):
    """
    Extracts the duration specified in the filename of audio files, trims the audio,
    and saves the trimmed version to the output directory with the same name.

    Parameters:
        input_dir (str or Path): Path to the directory containing the input audio files.
        output_dir (str or Path): Path to the directory where trimmed audio files will be saved.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Regular expression to extract start and end times
    duration_pattern = re.compile(r"___(\d+(?:\.\d+)?)\-(\d+(?:\.\d+)?)s___")

    for audio_file in input_dir.iterdir():
        if audio_file.suffix.lower() == ".wav":  # Process only WAV files
            match = duration_pattern.search(audio_file.stem)
            if match:
                start_time, end_time = map(float, match.groups())

                try:
                    # Open the WAV file
                    with wave.open(str(audio_file), "rb") as wav:
                        params = wav.getparams()
                        n_channels, sampwidth, framerate, n_frames = params[:4]

                        # Calculate start and end frames
                        start_frame = int(start_time * framerate)
                        end_frame = int(end_time * framerate)

                        # Set the position to start frame and read required frames
                        wav.setpos(start_frame)
                        frames = wav.readframes(end_frame - start_frame)

                        # Save trimmed audio
                        output_path = output_dir / audio_file.name
                        with wave.open(str(output_path), "wb") as output_wav:
                            output_wav.setparams((n_channels, sampwidth, framerate, 0, "NONE", "not compressed"))
                            output_wav.writeframes(frames)
                        print(f"Saved trimmed audio: {output_path}")
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
            else:
                print(f"Duration not found in filename: {audio_file.name}")

if __name__ == "__main__":
    # Define paths
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "data/audio/files"
    output_dir = base_dir / "data/preprocessed"

    # Extract and save trimmed audio
    extract_duration_and_save(input_dir, output_dir)
