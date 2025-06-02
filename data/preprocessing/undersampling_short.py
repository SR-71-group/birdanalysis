import os
from pydub import AudioSegment
from collections import defaultdict
from google.colab import drive

from pathlib import Path


# ❗ SCRIPT WAS USED FOR PREPROCESSING STEP WHICH RESULTED IN CLEAR DATASET ❗

base_dir = Path(__file__).resolve().parent
# Set input and output directories
input_dir = base_dir / 'data/clear'
output_dir = 'data/new_dataset'
os.makedirs(output_dir, exist_ok=True)

# Duration thresholds (in milliseconds)
min_duration = 2.5 * 1000  # 2.5 seconds in milliseconds
max_duration = 8 * 1000   # 10 seconds in milliseconds

# List of species codes to process
target_species = {'st', 're'}  # Song Thrush (st) and Redwing (re)

# Group files by species code (last two characters in filename)
species_files = defaultdict(list)
for filename in os.listdir(input_dir):
    if filename.endswith('.wav'):  # Assuming all files are .wav
        species_code = filename.split("___")[-1][:2]  # Extract last two characters
        if species_code in target_species:  # Only process st and re
            species_files[species_code].append(filename)

# Process each species group
for species_code, files in species_files.items():
    print(f"Processing species: {species_code}")

    # Load audio files and filter by duration
    short_clips = []
    for filename in files:
        filepath = os.path.join(input_dir, filename)
        audio = AudioSegment.from_file(filepath)
        if len(audio) < min_duration:
            short_clips.append((filename, audio))  # Store filename and audio

    # Concatenate short clips until they meet the required duration
    concatenated_clip = AudioSegment.empty()
    first_filename = None
    for filename, clip in short_clips:
        if not first_filename:  # Record the name of the first file
            first_filename = filename

        concatenated_clip += clip
        if len(concatenated_clip) >= min_duration:
            # Trim if it exceeds the maximum duration
            if len(concatenated_clip) > max_duration:
                concatenated_clip = concatenated_clip[:int(max_duration)]

            # Save the concatenated clip using the first file's name
            output_filename = first_filename  # Use the name of the first file
            output_filepath = os.path.join(output_dir, output_filename)
            concatenated_clip.export(output_filepath, format="wav")
            print(f"Saved: {output_filename}")

            # Reset for the next concatenated clip
            concatenated_clip = AudioSegment.empty()
            first_filename = None

    # Handle any remaining clip if it's between 2.5 and 10 seconds
    if first_filename and min_duration <= len(concatenated_clip) <= max_duration:
        output_filename = first_filename
        output_filepath = os.path.join(output_dir, output_filename)
        concatenated_clip.export(output_filepath, format="wav")
        print(f"Saved: {output_filename}")
