import numpy as np
import pandas as pd
from pathlib import Path
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import LabelEncoder
import os


def identify_rare_classes(csv_path, threshold=100):
    """Identify classes with fewer samples than the threshold."""
    df = pd.read_csv(csv_path)
    rare_classes = df['species_code'].value_counts()[lambda x: x < threshold].index.tolist()
    return df[df['species_code'].isin(rare_classes)], rare_classes


def load_spectrograms(rare_samples, spectrogram_dir, target_shape=(128, 128)):
    """Load spectrograms for rare samples into a list."""
    spectrogram_data = []
    labels = []
    metadata_rows = []

    for _, row in rare_samples.iterrows():
        file_name_without_extension = row['filename'].replace('.wav', '')  # Remove .wav part
        file_path = Path(spectrogram_dir) / f"{file_name_without_extension}.npy"
        if file_path.exists():
            spectrogram = np.load(file_path)

            # Resize or pad spectrogram to target_shape
            if spectrogram.shape != target_shape:
                resized_spectrogram = np.zeros(target_shape, dtype=spectrogram.dtype)
                min_rows = min(target_shape[0], spectrogram.shape[0])
                min_cols = min(target_shape[1], spectrogram.shape[1])
                resized_spectrogram[:min_rows, :min_cols] = spectrogram[:min_rows, :min_cols]
                spectrogram = resized_spectrogram

            spectrogram_data.append(spectrogram.flatten())  # Flatten for SMOTE/ADASYN
            labels.append(row['species_code'])
            metadata_rows.append(row)  # Keep original metadata
        else:
            print(f"File not found: {file_path}")

    return np.array(spectrogram_data), np.array(labels), metadata_rows


def apply_oversampling(data, labels, metadata_rows, method="SMOTE", min_samples=6):
    """Apply SMOTE or ADASYN to oversample rare classes."""
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Duplicate rare classes to meet minimum requirements for SMOTE/ADASYN
    augmented_data = data.tolist()
    augmented_labels = labels_encoded.tolist()
    augmented_metadata = metadata_rows.copy()
    class_counts = np.bincount(labels_encoded)

    for class_label, count in enumerate(class_counts):
        if count < min_samples:
            class_indices = np.where(labels_encoded == class_label)[0]
            num_duplicates = min_samples - count
            duplicates = np.random.choice(class_indices, num_duplicates, replace=True)

            augmented_data.extend(data[duplicates])
            augmented_labels.extend(labels_encoded[duplicates])
            augmented_metadata.extend([metadata_rows[idx] for idx in duplicates])

    # Apply SMOTE or ADASYN
    if method == "SMOTE":
        oversampler = SMOTE(k_neighbors=5)
    else:
        oversampler = ADASYN()

    X_resampled, y_resampled_encoded = oversampler.fit_resample(
        np.array(augmented_data), np.array(augmented_labels)
    )

    # Decode labels back to original strings
    y_resampled = label_encoder.inverse_transform(y_resampled_encoded)

    return X_resampled, y_resampled, augmented_metadata

def save_synthetic_spectrograms(X_resampled, y_resampled, metadata_rows, original_shape, output_dir):
    """Save synthetic spectrograms and return updated metadata rows."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    new_entries = []
    for data, label, metadata in zip(X_resampled, y_resampled, metadata_rows):
        reshaped_spectrogram = data.reshape(original_shape)

        # Update synthetic filename based on original metadata
        base_name, ext = os.path.splitext(metadata['filename'])

        # Insert "Syn" after "Tautenburg" and preserve the rest
        if "Tautenburg" in base_name:
            prefix, tautenburg_and_rest = base_name.split("Tautenburg", 1)
            if "Syn" not in tautenburg_and_rest:
                new_filename = f"{prefix}TautenburgSyn{tautenburg_and_rest}{ext}"
            else:
                new_filename = f"{prefix}Tautenburg{tautenburg_and_rest}{ext}"
        else:
            # For filenames without "Tautenburg," add "Syn" at the end
            new_filename = f"{base_name}Syn{ext}"

        # Save the spectrogram
        np.save(output_dir / new_filename, reshaped_spectrogram)

        # Copy original metadata and update filename
        synthetic_metadata = metadata.copy()
        synthetic_metadata['filename'] = new_filename
        new_entries.append(synthetic_metadata)

    return new_entries


def update_metadata(csv_path, new_entries, output_csv):
    """Update the metadata CSV with new synthetic samples, excluding unnecessary fields."""
    df = pd.read_csv(csv_path)
    df_new = pd.DataFrame(new_entries)

    # Remove the 'duration_category' column if it exists
    if 'duration_category' in df_new.columns:
        df_new = df_new.drop(columns=['duration_category'])

    df_updated = pd.concat([df, df_new], ignore_index=True)

    if 'duration_category' in df_updated.columns:
        df_updated = df_updated.drop(columns=['duration_category'])

    df_updated.to_csv(output_csv, index=False)
    print(f"Updated metadata saved to {output_csv}")


if __name__ == "__main__":
    # Paths and parameters
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "metadata/sl_win_annotations.csv"
    spectrogram_dir = base_dir / "mel-spectrograms"
    output_dir = base_dir / "mel-spectrograms"
    output_csv = base_dir / "metadata/oversampled_annotations.csv"
    rare_class_threshold = 100  # Updated threshold
    oversampling_method = "SMOTE"

    # Identify rare classes
    rare_samples, rare_classes = identify_rare_classes(csv_path, threshold=rare_class_threshold)
    print(f"Rare classes: {rare_classes}")

    # Load spectrograms for rare classes
    spectrogram_data, labels, metadata_rows = load_spectrograms(rare_samples, spectrogram_dir)

    if spectrogram_data.size > 0:
        # Apply oversampling
        X_resampled, y_resampled, resampled_metadata = apply_oversampling(
            spectrogram_data, labels, metadata_rows, method=oversampling_method
        )

        # Save synthetic spectrograms
        original_shape = spectrogram_data[0].reshape(-1, 128).shape  # Assuming 128 mel-bins
        new_entries = save_synthetic_spectrograms(
            X_resampled, y_resampled, resampled_metadata, original_shape, output_dir
        )

        # Update metadata
        update_metadata(csv_path, new_entries, output_csv)
    else:
        print("No spectrogram data loaded. Ensure the rare classes have corresponding files.")
