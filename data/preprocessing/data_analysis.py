#
# SCRIPT FOR ANALYSING DATASET FROM CSV FILE
# shows distribution of bird calls by species, frequencies and duration from filename
# however the durations are not reliable for clear dataset as the calls were altered
# to match the requirement of being under 10s long

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import species_mapping
from pathlib import Path

# File and dataset setup
# ❗COULD BE USED TO ANALYSE DATASET FROM ANY OF CSV FILEs❗
base_dir = Path(__file__).resolve().parent.parent
# CHOOSING OF CSV FILE for further analysis
data_file = base_dir / "metadata/undersampled_kmeans.csv"
df = pd.read_csv(data_file, header=None, names=["filename", "species_code"])

# Parse the species_mapping from dataset.py (assuming it exists and is imported)
species_dict = {species_mapping[i]: species_mapping[i + 1] for i in range(0, len(species_mapping), 3)}

# Map species code to species name
df['species_name'] = df['species_code'].map(species_dict)

# Function to extract frequencies, duration, and additional features
def extract_features(filename):
    parts = filename.split('___')
    if len(parts) < 4:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    freq_part = parts[2]  # '6302-9081kHz' or similar
    duration_part = parts[3]  # '10-10.6s' or similar

    # Extract frequency range (convert kHz to Hz)
    try:
        frequencies = freq_part.split('-')
        lower_freq = int(frequencies[0].replace('kHz', ''))   
        upper_freq = int(frequencies[1].replace('kHz', ''))   
    except ValueError:
        lower_freq = upper_freq = np.nan

    # Extract duration (convert to seconds)
    try:
        duration = duration_part.split('s')[0]  # Get the value before 's'
        duration_value = float(duration.split('-')[1])  # Take the duration value after '-'
    except (IndexError, ValueError):
        duration_value = np.nan

    # Calculate additional features
    bandwidth = upper_freq - lower_freq if not np.isnan(lower_freq) and not np.isnan(upper_freq) else np.nan
    central_freq = (lower_freq + upper_freq) / 2 if not np.isnan(lower_freq) and not np.isnan(upper_freq) else np.nan

    return lower_freq, upper_freq, duration_value, bandwidth, central_freq

# Apply the extraction function to the dataset
df[['lower_freq', 'upper_freq', 'duration', 'bandwidth', 'central_freq']] = df['filename'].apply(
    lambda x: pd.Series(extract_features(x))
)

# Total Calls
total_calls = len(df)
print(f"Total number of calls: {total_calls}")

# Calls per Species
calls_per_species = df['species_name'].value_counts()
print("\nNumber of calls per species:")
print(calls_per_species)

# Visualizations

# Species Count Bar Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=calls_per_species.index, y=calls_per_species.values, palette="viridis")
plt.title("Number of Calls per Species", fontsize=16)
plt.xlabel("Species", fontsize=12)
plt.ylabel("Number of Calls", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Frequency Bandwidth Distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['bandwidth'], bins=30, kde=True, color='purple')
plt.title("Frequency Bandwidth Distribution", fontsize=16)
plt.xlabel("Bandwidth (Hz)", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# Central Frequency Distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['central_freq'], bins=30, kde=True, color='blue')
plt.title("Central Frequency Distribution", fontsize=16)
plt.xlabel("Central Frequency (Hz)", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# Boxplot of Frequencies by Species
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='species_name', y='bandwidth', palette="Set2")
plt.title("Bandwidth Distribution by Species", fontsize=16)
plt.xlabel("Species", fontsize=12)
plt.ylabel("Bandwidth (Hz)", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Duration vs Central Frequency (Scatterplot)
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='central_freq', y='duration', hue='species_name', palette="Set1")
plt.title("Duration vs Central Frequency by Species", fontsize=16)
plt.xlabel("Central Frequency (Hz)", fontsize=12)
plt.ylabel("Duration (s)", fontsize=12)
plt.legend(title="Species")
plt.grid(True)
plt.tight_layout()
plt.show()

