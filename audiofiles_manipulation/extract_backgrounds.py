import os
from pathlib import Path
from pydub import AudioSegment

def parse_filename(file_name):
    """
    Parse the filename to extract metadata including the time range to cut out.

    Example:
        2459855.447247_Tautenburg___6503-9785kHz___10-10.7s___st.wav
        Returns: {
            "prefix": "2459855.447247_Tautenburg___6503-9785kHz",
            "start_time": 10.0,
            "end_time": 10.7,
            "suffix": "st"
        }
    """
    parts = file_name.split("___")
    prefix = "___".join(parts[:2])  # Combine first two parts for prefix
    time_range = parts[2].replace("s", "").split("-")
    suffix = parts[3].replace(".wav", "")

    start_time = float(time_range[0])
    end_time = float(time_range[1])
    return {"prefix": prefix, "start_time": start_time, "end_time": end_time, "suffix": suffix}

def cut_and_save_background(input_file, output_dir):
    """
    Process an audio file, remove the specified range, and save the rest as background audio.

    Parameters:
        input_file (Path or str): Path to the input .wav file.
        output_dir (Path or str): Directory where the processed background file will be saved.
    """
    file_name = os.path.basename(input_file)
    parsed_info = parse_filename(file_name)

    # Load the audio file
    audio = AudioSegment.from_wav(input_file)
    audio_duration = len(audio) / 1000  # Convert duration to seconds

    # Convert start and end times to milliseconds
    start_ms = int(parsed_info["start_time"] * 1000)
    end_ms = int(parsed_info["end_time"] * 1000)

    # Cut out the specified segment
    before_segment = audio[:start_ms]
    after_segment = audio[end_ms:]
    background_audio = before_segment + after_segment

    # Generate new filename
    new_duration = len(background_audio) / 1000  # Duration of background audio in seconds
    new_filename = f"{parsed_info['prefix']}___0-{int(new_duration)}s___background.wav"

    # Save the background audio
    output_path = Path(output_dir) / new_filename
    background_audio.export(output_path, format="wav")
    print(f"Processed and saved: {output_path}")

def main():
    """Main function to process all audio files."""
    # Define directories
    script_dir = Path(__file__).resolve().parent
    audio_dir = script_dir / 'data' / 'audio' / 'files'
    output_dir = script_dir / 'data' / 'audio' /'backgrounds'

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each .wav file in the audio directory
    for file_name in os.listdir(audio_dir):
        if file_name.endswith('.wav'):
            input_file = audio_dir / file_name
            cut_and_save_background(input_file, output_dir)

if __name__ == "__main__":
    main()
