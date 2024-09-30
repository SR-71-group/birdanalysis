# file_saver.py
from pathlib import Path

def save_audio(audio, filename, output_dir):
    """Save the processed audio to the target directory with the correct filename."""
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Construct the output path using Path object
    output_path = Path(output_dir) / filename
    audio.export(output_path, format="wav")
    return output_path

def generate_background_filename(parsed_info):
    """Generate the new filename for the background noise file."""
    # Use the parsed information, but set the timing to 0-2min30s and species to 'background'
    return f"{parsed_info['julian_date']}_{parsed_info['location']}___{parsed_info['frequency_range']}___0-2m30s___background.wav"
