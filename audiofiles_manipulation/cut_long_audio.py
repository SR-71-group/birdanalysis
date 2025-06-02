import pandas as pd
import os
from pathlib import Path
import soundfile as sf


def extract_duration_from_filename(filename):
    """
    Extracts the start and end times from the filename and calculates the duration.
    """
    try:
        parts = filename.split("___")
        duration_range = parts[2].replace("s", "").split("-")
        start_time = float(duration_range[0])
        end_time = float(duration_range[1])
        return end_time - start_time
    except (IndexError, ValueError) as e:
        print(f"Error processing filename: {filename}. Error: {e}")
        return None


def process_calls_by_duration_category(annotations_path, audio_dir, output_dir, updated_annotations_path):
    # Load annotations
    df = pd.read_csv(annotations_path)

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize list for updated annotations
    updated_records = []

    for _, row in df.iterrows():
        species = row["species_code"]
        filename = row["filename"]

        # Calculate duration from filename
        duration = extract_duration_from_filename(filename)
        if duration is None:
            print(f"Skipping file due to invalid duration: {filename}")
            continue

        # Load the audio file
        audio_path = Path(audio_dir) / filename
        if not audio_path.exists():
            print(f"File not found: {audio_path}")
            continue

        try:
            audio, sr = sf.read(audio_path)
        except Exception as e:
            print(f"Error loading file {audio_path}: {e}")
            continue

        # Handle files based on duration
        if duration > 120:  # Over 2 minutes
            num_parts = 3
        elif duration > 61:  # Between 1 and 2 minutes
            num_parts = 2
        else:  # Keep files under 1 minute without changes
            updated_records.append(row)
            continue

        # Calculate segment duration
        segment_duration = duration / num_parts

        for i in range(num_parts):
            # Calculate new start and end times
            start_time = i * segment_duration
            end_time = (i + 1) * segment_duration

            # Update the filename with the new time range
            original_parts = filename.split("___")
            if len(original_parts) < 4:
                print(f"Invalid filename format: {filename}")
                continue
            time_range = f"{start_time:.1f}-{end_time:.1f}s"
            new_filename = f"{original_parts[0]}___{original_parts[1]}___{time_range}___{species}.wav"
            new_file_path = output_dir / new_filename

            # Save the new audio segment
            start_sample = int(i * segment_duration * sr)
            end_sample = int((i + 1) * segment_duration * sr)
            sf.write(new_file_path, audio[start_sample:end_sample], sr)

            # Add a new record for this segment
            new_row = row.copy()
            new_row["filename"] = new_filename
            updated_records.append(new_row)

        # Do not add the original long call to the final dataset
        print(f"Original long call {filename} split into {num_parts} parts and excluded from the final dataset.")

    # Create a new annotations DataFrame
    updated_df = pd.DataFrame(updated_records)

    # Save the updated annotations file
    updated_df.to_csv(updated_annotations_path, index=False)
    print(f"Updated annotations saved to {updated_annotations_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    annotations_path = base_dir / "data/metadata/undersampled_annotations.csv"  # Adjust as needed
    audio_dir = base_dir / "data/preprocessed"  # Parent directory for original audio files
    output_dir = base_dir / "data/preprocessed"  # Directory for split audio files
    updated_annotations_path = base_dir / "data/metadata/cut_long_calls_annotations.csv"  # Output annotations file

    process_calls_by_duration_category(annotations_path, audio_dir, output_dir, updated_annotations_path)
