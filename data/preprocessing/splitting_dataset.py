import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
# ❗ SCRIPT COULD BE USED FOR SOME MODELS WHICH REQUIRE PRECEDING SPLIT❗
def stratified_split(data, test_size=0.15, val_size=0.15, min_samples=6):
    """
    Stratified splitting of data into train, validation, and test sets.

    Args:
        data (pd.DataFrame): Dataset with columns ["filename", "species_code", ...].
        test_size (float): Proportion of data to reserve for testing.
        val_size (float): Proportion of data to reserve for validation.
        min_samples (int): Minimum number of samples for a class to be included.

    Returns:
        dict: Containing train, validation, and test datasets.
    """
    # Drop classes with fewer than `min_samples`
    class_counts = data['species_code'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    filtered_data = data[data['species_code'].isin(valid_classes)]

    train_data, temp_data = train_test_split(
        filtered_data, stratify=filtered_data['species_code'], test_size=test_size + val_size, random_state=42
    )
    val_ratio = val_size / (test_size + val_size)  # Adjust validation proportion
    val_data, test_data = train_test_split(
        temp_data, stratify=temp_data['species_code'], test_size=1 - val_ratio, random_state=42
    )

    return {
        "train": train_data,
        "validation": val_data,
        "test": test_data
    }

if __name__ == "__main__":
    # Load your dataset (update with your file path)
    base_dir = Path(__file__).resolve().parent.parent
    dataset_path = base_dir / "metadata/spectrograms_dataset.csv"
    data = pd.read_csv(dataset_path)

    # Perform stratified split
    split_data = stratified_split(data, test_size=0.15, val_size=0.15, min_samples=10)

    # Save the splits
    split_data['train'].to_csv(base_dir / "metadata/train_dataset.csv", index=False)
    split_data['validation'].to_csv(base_dir / "metadata/validation_dataset.csv", index=False)
    split_data['test'].to_csv(base_dir / "metadata/test_dataset.csv", index=False)

    print(f"Train size: {len(split_data['train'])}")
    print(f"Validation size: {len(split_data['validation'])}")
    print(f"Test size: {len(split_data['test'])}")
