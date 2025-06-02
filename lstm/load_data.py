import pandas as pd
import numpy as np
from pathlib import Path

def load_data(train_csv, test_csv, val_csv, spectrogram_dir):
    def load_from_csv(csv_file):
        """Load data from a single CSV."""
        df = pd.read_csv(csv_file)
        X_data, y_labels, species_map = [], [], {}

        # Create species mapping
        species_map = {species: idx for idx, species in enumerate(df['species_code'].unique())}
        df['species_label'] = df['species_code'].map(species_map)

        # Load spectrograms
        for _, row in df.iterrows():
            filename = row['filename']
            label = row['species_label']
            spec_path = Path(spectrogram_dir) / f"{filename}.npy"
            if spec_path.exists():
                spectrogram = np.load(spec_path)
                X_data.append(spectrogram)
                y_labels.append(label)
            else:
                print(f"Missing spectrogram file: {spec_path}")

        # Pad spectrograms
        max_rows = max(x.shape[0] for x in X_data)
        max_cols = max(x.shape[1] for x in X_data)
        padded_spectrograms = [
            np.pad(x, ((0, max_rows - x.shape[0]), (0, max_cols - x.shape[1])), mode='constant').T
            for x in X_data
        ]
        return np.array(padded_spectrograms), np.array(y_labels), species_map

    # Load datasets
    train_data = load_from_csv(train_csv)
    test_data = load_from_csv(test_csv)
    val_data = load_from_csv(val_csv) if val_csv.exists() else None
    return train_data, test_data, val_data
