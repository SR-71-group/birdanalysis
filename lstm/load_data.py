import pandas as pd
import numpy as np
from pathlib import Path

def load_data(train_csv, test_csv, spectrogram_dir):
    """Load training and testing data from CSV files and corresponding spectrograms."""

    def load_from_csv(csv_file):
        """Helper function to load data from a single CSV file."""
        # Construct the full path to the CSV file
        csv_path = Path(__file__).resolve().parent.parent / 'data' / 'metadata' / csv_file
        df = pd.read_csv(csv_path)

        X_data = []
        y_labels = []

        # Map species codes to integers
        species_map = {species: idx for idx, species in enumerate(df['species_code'].unique())}
        df['species_label'] = df['species_code'].map(species_map)

        for index, row in df.iterrows():
            filename = row['filename']  # Get the filename from the CSV
            species_label = row['species_label']

            # Construct the spectrogram file path
            spectrogram_path = Path(spectrogram_dir) / f"{filename}_spectrogram.npy"  # Append _spectrogram.npy

            # Debugging output to check the spectrogram path
            print(f"Looking for spectrogram: {spectrogram_path}")

            if spectrogram_path.exists():
                spectrogram = np.load(spectrogram_path)
                spectrogram_reshaped = spectrogram.T  # Ensure correct shape for LSTM
                X_data.append(spectrogram_reshaped)
                y_labels.append(species_label)
            else:
                print(f"Warning: Spectrogram file does not exist: {spectrogram_path}")

        print(f"Loaded {len(X_data)} samples from {csv_file}.")  # Summary of loaded data
        return np.array(X_data), np.array(y_labels), species_map

    # Load training data
    X_train, y_train, train_species_map = load_from_csv(train_csv)

    # Load testing data
    X_test, y_test, test_species_map = load_from_csv(test_csv)

    return (X_train, y_train, train_species_map), (X_test, y_test, test_species_map)
