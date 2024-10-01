from load_data import load_data
from train_model import train_lstm_model
from pathlib import Path

def main():
    # Get the root directory where the script is being executed
    root_dir = Path(__file__).resolve().parent.parent  # Navigate up two levels to the project root
    train_csv_filename = root_dir / 'data' / 'metadata' / 'annotations_train_resampled.csv'  # Correctly construct path
    test_csv_filename = root_dir / 'data' / 'metadata' / 'annotations_test.csv'  # Ensure the path is correct
    spectrogram_dir = root_dir / 'data' / 'spectrograms'  # Relative path to the spectrograms

    # Debug prints to verify paths
    print(f"Training CSV path: {train_csv_filename}")
    print(f"Testing CSV path: {test_csv_filename}")
    print(f"Spectrogram directory: {spectrogram_dir}")

    # Check if the files exist before loading
    if not train_csv_filename.exists():
        print(f"Error: Training CSV not found at {train_csv_filename}")
        return
    if not test_csv_filename.exists():
        print(f"Error: Testing CSV not found at {test_csv_filename}")
        return
    if not spectrogram_dir.exists():
        print(f"Error: Spectrogram directory not found at {spectrogram_dir}")
        return

    # Load data from CSV files
    (X_train, y_train, train_species_map), (X_test, y_test, test_species_map) = load_data(
        train_csv_filename.name,
        test_csv_filename.name,
        spectrogram_dir
    )

    # Check if all the required spectrogram files exist
    missing_files = False
    for i, spectrogram_path in enumerate(X_train):
        if not Path(spectrogram_path).exists():
            print(f"Warning: Training spectrogram file does not exist: {spectrogram_path}")
            missing_files = True
    for i, spectrogram_path in enumerate(X_test):
        if not Path(spectrogram_path).exists():
            print(f"Warning: Testing spectrogram file does not exist: {spectrogram_path}")
            missing_files = True

    # If any spectrograms are missing, we should abort
    if missing_files:
        print("Error: Missing spectrogram files. Please ensure all required files are present.")
        return

    # Train the model with the loaded data
    train_lstm_model(X_train, y_train, train_species_map)

    # Optionally evaluate the model on test data
    # evaluate_model(test_model, X_test, y_test)

if __name__ == "__main__":
    main()
