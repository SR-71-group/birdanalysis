from clearml import Logger, Task

from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import Sequential
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.python.keras.callbacks import LearningRateScheduler
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical

from tensorflow.python.keras.callbacks import Callback
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from tensorflow.keras.layers import Bidirectional
import tensorflow as tf

def check_gpu():
    """
    Check if TensorFlow is running on GPU and log the available devices.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Running on GPU: {tf.config.experimental.get_device_details(gpus[0])['device_name']}")
    else:
        print("Running on CPU. No GPU detected.")


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam



def build_lstm_model(input_shape, num_classes):
    # Create a Sequential model
    model = Sequential()

    # Add layers
    model.add(Bidirectional(LSTM(256, return_sequences=True, input_shape=input_shape)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64, return_sequences=False)))

    # Add Dense layers
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model





def train_lstm_model(train_data, test_data, val_data):
    # Check if GPU is available
    check_gpu()

    # Unpack datasets
    X_train, y_train, species_map = train_data
    X_test, y_test, _ = test_data
    if val_data:
        X_val, y_val, _ = val_data
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Normalize data
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

    # One-hot encode labels
    num_classes = len(species_map)
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(np.argmax(y_train, axis=1)),
        y=np.argmax(y_train, axis=1)
    )
    class_weights = {i: weight for i, weight in enumerate(class_weights)}

    # Build model
    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        num_classes=num_classes
    )
    # Callbacks
    clearml_callback = CustomMetricsCallback(species_map=species_map, validation_data=(X_val, y_val))

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=20,  # Increased epochs to account for removal of early stopping
        batch_size=64,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=clearml_callback,
    )

    # Test evaluation
    test_predictions = np.argmax(model.predict(X_test), axis=1)
    test_true_labels = np.argmax(y_test, axis=1)

    # Log metrics
    cm = confusion_matrix(test_true_labels, test_predictions)
    classification_report_str = classification_report(test_true_labels, test_predictions)

    logger = Logger.current_task().get_logger()
    logger.report_scalar("Accuracy", "Test", iteration=0, value=np.mean(test_predictions == test_true_labels))
    logger.report_confusion_matrix(
        title="Confusion Matrix",
        series="Test",
        iteration=0,
        matrix=cm.tolist(),
        xlabels=list(species_map.keys()),
        ylabels=list(species_map.keys()),
    )

    return model, history, species_map


