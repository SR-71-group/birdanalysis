import pandas as pd
import os
from pathlib import Path
from sklearn.cluster import KMeans
from collections import Counter

# ❗ SCRIPT WAS USED FOR PREPROCESSING STEP WHICH RESULTED IN CLEAR DATASET ❗

def simple_audio_features(audio_path):
    """
    Simple feature extraction using raw wave analysis.
    Converts Path objects to strings before using wave.open().
    """
    import wave
    if not isinstance(audio_path, str):
        audio_path = str(audio_path)  # Convert Path to string

    with wave.open(audio_path, 'rb') as wav_file:
        frames = wav_file.readframes(-1)
        audio = [int.from_bytes(frames[i:i+2], 'little', signed=True) for i in range(0, len(frames), 2)]
        return [sum(audio) / len(audio)]  # Example: mean value of audio samples




def identify_top_classes(df, top_n=2):
    """
    Identify the most common classes in the dataset.
    """
    class_counts = Counter(df['species_code'])
    return [item[0] for item in class_counts.most_common(top_n)]

def undersample_class(df, class_name, audio_dir, target_size=500):
    """
    Undersample a specific class using clustering to retain diverse records.
    """
    class_records = df[df['species_code'] == class_name]
    features = []
    indices = []

    for _, row in class_records.iterrows():
        audio_path = audio_dir / row['filename']
        try:
            feature = simple_audio_features(audio_path)
            features.append(feature)
            indices.append(row.name)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

    if len(features) > target_size:
        kmeans = KMeans(n_clusters=target_size, random_state=42)
        kmeans.fit(features)
        labels = kmeans.labels_

        # Select one record per cluster
        selected_indices = []
        for cluster in range(target_size):
            cluster_indices = [idx for idx, lbl in zip(indices, labels) if lbl == cluster]
            if cluster_indices:
                selected_indices.append(cluster_indices[0])
    else:
        # Keep all records if less than or equal to target_size
        selected_indices = indices

    return selected_indices

def save_filtered_data(df, selected_indices, output_csv_path):
    """
    Save the new dataset after undersampling.
    """
    filtered_df = df.loc[selected_indices]
    filtered_df.to_csv(output_csv_path, index=False)
    print(f"Filtered data saved to {output_csv_path}")

if __name__ == "__main__":
    # Paths
    base_dir = Path(__file__).resolve().parent

    annotations_path = base_dir / "metadata" / "fulldataset.csv"
    audio_dir = base_dir / "clear"
    output_csv_path = base_dir / "metadata" / "undersampled_annotations.csv"

    # Load annotations
    df = pd.read_csv(annotations_path)

    # Identify the top 2 most common classes
    top_classes = identify_top_classes(df, top_n=2)

    # Retain all other classes and undersample top 2
    selected_indices = []
    for class_name in top_classes:
        selected_indices += undersample_class(df, class_name, audio_dir, target_size=500)

    # Add the rest of the data (unchanged)
    rare_class_indices = df[~df['species_code'].isin(top_classes)].index.tolist()
    selected_indices += rare_class_indices

    # Save the new dataset
    save_filtered_data(df, selected_indices, output_csv_path)
