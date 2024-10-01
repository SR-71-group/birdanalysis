
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from clearml import Task

from clearml_tracking.custom_metrix_callback import CustomMetricsCallback


def build_lstm_model(input_shape, num_classes):
    """Build and compile the LSTM model."""
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
def train_lstm_model(X_data, y_labels, species_map):
    """Train the LSTM model on the provided data."""

    # Initialize ClearML task
    task = Task.init(project_name="Bird Call Detection", task_name="LSTM Spectrogram Training")

    # Prepare labels (one-hot encoding)
    y_labels_one_hot = to_categorical(y_labels, num_classes=len(species_map))

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_labels_one_hot, test_size=0.2, random_state=42)

    # Build model
    model = build_lstm_model(input_shape=(X_data.shape[1], X_data.shape[2]), num_classes=len(species_map))
    model.summary()  # Optional: print model summary

    # Custom Metrics Callback for per-species accuracy
    custom_metrics_callback = CustomMetricsCallback(species_map)

    # Train model and log metrics
    history = model.fit(
        X_train, y_train,
        epochs=20, batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[custom_metrics_callback]  # Add the custom callback here
    )

    # Evaluate the model
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_acc}")
