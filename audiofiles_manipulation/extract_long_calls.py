import os
import shutil
import pandas as pd
from pathlib import Path

def filter_and_copy_long_calls(annotations_file, audio_dir, output_dir):
    """
    Filter audio files with 'duration_category = over 2min' from annotations_df.csv
    and copy them to a specified directory for manual review.

    Parameters:
    - annotations_file (str or Path): Path to the CSV file with annotations.
    - audio_dir (str or Path): Path to the directory containing audio files.
    - output_dir (str or Path): Path to the directory where long calls will be copied.
    """
    # Ensure paths are Path objects
    annotations_file = Path(annotations_file)
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the annotations file
    try:
        annotations_df = pd.read_csv(annotations_file)
    except FileNotFoundError:
        print(f"Error: Annotations file not found at {annotations_file}")
        return

    # Filter rows where duration_category == 'over 2min'
    long_calls_df = annotations_df[annotations_df['duration_category'] == 'Over 2 min']

    if long_calls_df.empty:
        print("No files found with 'duration_category = over 2min'.")
        return

    # Get the filenames for long calls
    long_call_files = long_calls_df['filename'].values

    # Copy files to the output directory
    copied_files = []
    for file_name in long_call_files:
        source_file = audio_dir / file_name
        if source_file.exists():
            destination_file = output_dir / file_name
            shutil.copy(source_file, destination_file)
            copied_files.append(file_name)
            print(f"Copied: {file_name} to {output_dir}")
        else:
            print(f"Warning: File not found {source_file}")

    print("\nSummary:")
    print(f"Total files marked as 'over 2min': {len(long_call_files)}")
    print(f"Files successfully copied: {len(copied_files)}")
    print(f"Files missing from audio directory: {len(long_call_files) - len(copied_files)}")

if __name__ == "__main__":
    # Define paths relative to the current script
    base_dir = Path(__file__).resolve().parent.parent  # Move up one directory level
    annotations_csv = base_dir / 'data/metadata/undersampled_annotations.csv'  # Path to annotations file
    audio_directory = base_dir / 'data/preprocessed'  # Path to audio files
    output_directory = base_dir / 'data/audio/longcalls2'  # Path to store long calls

    # Run the filter and copy function
    filter_and_copy_long_calls(annotations_csv, audio_directory, output_directory)
