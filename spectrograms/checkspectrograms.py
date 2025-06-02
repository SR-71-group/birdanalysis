import pandas as pd
from pathlib import Path

def check_spectrogram_files(csv_file, spectrogram_dir):
    # Load the CSV file
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file not found: {csv_file}")
        return

    # Extract filenames from the CSV
    # Ensure we preserve the full filename with all its parts
    if 'filename' not in df.columns:
        print(f"Error: 'filename' column not found in {csv_file}")
        return

    filenames = df['filename'].values

    # Append "_spectrogram.npy" to the filenames
    spectrogram_paths = [spectrogram_dir / (f + '_spectrogram.npy') for f in filenames]

    # Check if the corresponding spectrogram files exist
    found_files = []
    missing_files = []

    for f in spectrogram_paths:
        if f.exists():
            found_files.append(f)
        else:
            missing_files.append(f)

    # Print the results
    print(f"Total files in CSV: {len(filenames)}")
    print(f"Spectrogram files found: {len(found_files)}")
    print(f"Spectrogram files missing: {len(missing_files)}")

    if missing_files:
        print("\nMissing spectrogram files:")
        for mf in missing_files:
            print(mf)

if __name__ == "__main__":
    # Specify the paths
    csv_file = Path('data/metadata/annotations_test.csv')  # Adjust as needed
    spectrogram_dir = Path('data/padded_spectrograms')  # Adjust as needed

    # Check the spectrogram files
    check_spectrogram_files(csv_file, spectrogram_dir)
