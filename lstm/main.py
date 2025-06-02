from clearml import Task
from pathlib import Path
import os

from load_data import load_data
from train_model import train_lstm_model
import tensorflow as tf

def main():
    task = Task.init(
        project_name="Bird Call Detection",
        task_name="LSTM on Spectrograms",

    )
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    # Paths
    root_dir = Path(__file__).resolve().parent.parent  # Navigate up two levels
    train_csv = root_dir / 'data' / 'metadata' / 'train_dataset.csv'
    test_csv = root_dir / 'data' / 'metadata' / 'test_dataset.csv'
    val_csv = root_dir / 'data' / 'metadata' / 'validation_dataset.csv'
    spectrogram_dir = root_dir / 'data' / 'mel-spectrograms'

    # Check file existence
    for path in [train_csv, test_csv, spectrogram_dir]:
        if not path.exists():
            print(f"Error: Missing file or directory at {path}")
            return


    print("Built with CUDA:", tf.test.is_built_with_cuda())
    print("Available GPUs:", tf.config.list_physical_devices('GPU'))
    # Load data
    datasets = load_data(train_csv, test_csv, val_csv, spectrogram_dir)


    # Train the model
    model, history, species_map = train_lstm_model(*datasets)

    # Save the trained model in Keras format
    model_path = root_dir / 'trained_lstm_model.keras'
    model.save(model_path)
    task.upload_artifact(name="Trained LSTM Model", artifact_object=model_path)

if __name__ == "__main__":
    main()
