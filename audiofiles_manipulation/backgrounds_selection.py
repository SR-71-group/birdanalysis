# file_filter.py
import os
from pathlib import Path
from pydub import AudioSegment

def get_audio_duration(file_path):
    """Returns the duration of an audio file in milliseconds."""
    audio = AudioSegment.from_wav(file_path)
    return len(audio)

def filter_longest_files(directory, top_n=1000):
    """
    Filters and returns the top_n longest files from the specified directory.

    Parameters:
        directory (Path or str): The directory where the audio files are stored.
        top_n (int): Number of longest files to return.

    Returns:
        List of file paths representing the top_n longest files.
    """
    files_with_duration = []

    # Loop through all .wav files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith('.wav'):
            file_path = Path(directory) / file_name
            duration_ms = get_audio_duration(file_path)
            # Append the (file_path, duration) to the list
            files_with_duration.append((file_path, duration_ms))

    # Sort files by duration (longest first)
    sorted_files = sorted(files_with_duration, key=lambda x: x[1], reverse=True)

    # Select the top_n longest files
    longest_files = sorted_files[:top_n]

    # Extract just the file paths from the sorted list
    longest_file_paths = [file_path for file_path, _ in longest_files]
    return longest_file_paths

def delete_unselected_files(directory, selected_files):
    """
    Deletes files from the directory that are not in the selected_files list.

    Parameters:
        directory (Path or str): The directory where the audio files are stored.
        selected_files (list of Path): The list of file paths that should be kept.
    """
    # Convert selected files to a set for faster lookups
    selected_files_set = set(selected_files)

    # Loop through all .wav files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith('.wav'):
            file_path = Path(directory) / file_name
            # If the file is not in the selected_files list, delete it
            if file_path not in selected_files_set:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
