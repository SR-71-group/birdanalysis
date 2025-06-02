import os
import torch
import torchaudio
import pandas as pd
from transformers import AutoFeatureExtractor
# ❗❗
# download clear dataset first from google drive repository
# ❗❗
extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
SAMPLE_RATE = 16000
base_dir = Path(__file__).resolve().parent
CSV_PATH = base_dir / "data/metadata/undersampled_kmeans.csv"
AUDIO_DIR = base_dir / "data/clear"
OUTPUT_DIR = base_dir / "precomputed_ast"

os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(CSV_PATH)

for i, row in df.iterrows():
    try:
        path = os.path.join(AUDIO_DIR, row['filename'])
        waveform, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)

        waveform = waveform.mean(dim=0)
        if waveform.shape[0] < SAMPLE_RATE // 2:
            continue  # skip too short

        inputs = extractor(
            waveform.numpy(),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=1024,
            truncation=True,
        )['input_values'].squeeze(0)

        # Save tensor & label as a tuple
        label = row['species_code']
        torch.save((inputs, label), os.path.join(OUTPUT_DIR, f"{row['filename']}.pt"))

    except Exception as e:
        print(f"Failed {row['filename']}: {e}")
