import os
import pandas as pd
import re
from audiofiles_manipulation.audio_parser import parse_filename

from pathlib import Path

# SCRIPT FOR CREATION OF CSV FILE FROM SELECTED FOLDER WITH FILES

def generate_csv(audio_dir, output_csv):
    """
    Generate a CSV file containing filename and species_code information.
    """
    rows = []

    # Iterate over all files in the directory
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.pt'):  # Adjust this to your audio file extension or .wav for raw audios
                try:

                    parsed_info = parse_filename(file)
                    species_code = parsed_info.get('species_code', 'unknown')
                    rows.append({'filename': file, 'species_code': species_code})
                except Exception as e:
                    print(f"Error parsing file {file}: {e}")

    # Create and save the CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"CSV saved to {output_csv}")


# ❗CHOOSING FOLDER WITH DATASET AND TO WHERE NEW CSV FILE SHOULD BE SAVED ❗
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    input_dir = base_dir / "precomputed_ast"
    output_dir = base_dir / "data/metadata/ast_dataset.csv"
    generate_csv(input_dir, output_dir)
