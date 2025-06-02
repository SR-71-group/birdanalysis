import pandas as pd
from pathlib import Path
import soundfile as sf

def augment_rare_calls_with_sliding_window(
        annotations_path, audio_dir, output_dir, updated_annotations_path, rare_threshold=50, window_size=5, overlap=0.5):
    """
    Augment rare calls using a sliding window approach.

    Args:
        annotations_path (str): Path to the input annotations CSV file.
        audio_dir (str): Directory containing the audio files.
        output_dir (str): Directory to save the augmented audio files.
        updated_annotations_path (str): Path to save the updated annotations CSV file.
        rare_threshold (int): Threshold for defining rare classes (default: 50 calls).
        window_size (float): Duration of each sliding window (in seconds).
        overlap (float): Overlap between consecutive windows (0 to 1).
    """
    # Load annotations
    df = pd.read_csv(annotations_path)

    # Identify rare species
    species_counts = df["species_code"].value_counts()
    rare_species = species_counts[species_counts < rare_threshold].index.tolist()

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize updated records
    updated_records = []

    for _, row in df.iterrows():
        species = row["species_code"]
        filename = row["filename"]

        # Skip non-rare classes
        if species not in rare_species:
            updated_records.append(row)
            continue

        # Load audio file
        audio_path = Path(audio_dir) / filename
        if not audio_path.exists():
            print(f"File not found: {audio_path}")
            continue

        try:
            # Extract start and end times from filename
            try:
                time_part = filename.split("___")[-2]
                start_time, end_time = map(float, time_part[:-1].split("-"))
            except (IndexError, ValueError) as e:
                print(f"Error parsing times from filename {filename}: {e}")
                continue

            audio, sr = sf.read(audio_path)
            duration = len(audio) / sr

            # Calculate sliding window parameters
            step_size = window_size * (1 - overlap)
            num_windows = int((end_time - start_time - window_size) / step_size) + 1

            for i in range(num_windows):
                # Calculate start and end times for the segment
                segment_start = start_time + i * step_size
                segment_end = segment_start + window_size
                if segment_end > end_time:
                    break

                # Update the filename with new time range
                time_range = f"{segment_start:.1f}-{segment_end:.1f}s"
                new_filename = f"{filename.rsplit('___', 1)[0]}___{time_range}___{species}.wav"
                new_file_path = output_dir / new_filename

                # Save the audio segment
                start_sample = int(segment_start * sr)
                end_sample = int(segment_end * sr)
                sf.write(new_file_path, audio[start_sample:end_sample], sr)

                # Add a new record for this segment
                new_row = row.copy()
                new_row["filename"] = new_filename
                updated_records.append(new_row)

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

    # Save updated annotations
    updated_df = pd.DataFrame(updated_records)
    updated_df.to_csv(updated_annotations_path, index=False)
    print(f"Augmented annotations saved to {updated_annotations_path}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    annotations_path = base_dir / "metadata/cut_long_calls_annotations.csv"  # Input annotations
    audio_dir = base_dir / "preprocessed"  # Original audio files directory
    output_dir = base_dir / "preprocessed"  # Directory for augmented audio files
    updated_annotations_path = base_dir / "metadata/sliding_window_annotations.csv"  # Updated annotations file

    augment_rare_calls_with_sliding_window(
        annotations_path,
        audio_dir,
        output_dir,
        updated_annotations_path,
        rare_threshold=80,  # Threshold for rare species
        window_size=5,  # Window size in seconds
        overlap=0.5  # Overlap between windows (50%)
    )
