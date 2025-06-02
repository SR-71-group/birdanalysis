import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix

import pickle
from pathlib import Path

# Parameters
SAMPLE_RATE = 16000 # should be higher
N_CLASSES = 0  # Dynamically updated
BATCH_SIZE = 16
EPOCHS = 20
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "clear"
ANNOTATIONS_FILE = BASE_DIR / "metadata/undersampled_kmeans.csv"
TARGET_SHAPE = (84, 84)  # Spectrogram dimensions supposed to be higher for more detailed description of dataset

# Step 1: Load annotations and filter valid files
annotations = pd.read_csv(ANNOTATIONS_FILE)
annotations['filepath'] = annotations['filename'].apply(lambda x: os.path.join(DATA_DIR, x))
annotations = annotations[annotations['filepath'].apply(os.path.isfile)]  # Filter missing files

# Step 2: Encode labels
label_encoder = LabelEncoder()
annotations['encoded_label'] = label_encoder.fit_transform(annotations['species_code'])
class_names = label_encoder.classes_
N_CLASSES = len(class_names)



# Step 3: Balance classes by oversampling underrepresented ones
class_counts = annotations['encoded_label'].value_counts()
max_class_size = class_counts.max()

balanced_dfs = []
for label in class_counts.index:
    class_subset = annotations[annotations['encoded_label'] == label]
    if len(class_subset) < max_class_size:
        class_subset = resample(
            class_subset,
            replace=True,
            n_samples=max_class_size,
            random_state=42
        )
    balanced_dfs.append(class_subset)

balanced_annotations = pd.concat(balanced_dfs).reset_index(drop=True)

print(f"Original dataset size: {len(annotations)}, Balanced dataset size: {len(balanced_annotations)}")

# Step 4: Train-validation-test split
train_df, temp_df = train_test_split(
    balanced_annotations,
    test_size=0.3,
    stratify=balanced_annotations['encoded_label'],
    random_state=42
)
valid_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df['encoded_label'],
    random_state=42
)

print(f"Train size: {len(train_df)}, Validation size: {len(valid_df)}, Test size: {len(test_df)}")
# sizes of spectrograms were reduced due to lack of computing powers :(
def compute_cqt(filepath, target_shape=(84, 84), n_bins=84, bins_per_octave=12, sample_rate=16000):
    try:
        audio, _ = librosa.load(filepath, sr=sample_rate, mono=True)

        # Dynamically calculate the required n_fft
        required_n_fft = 2 ** int(np.ceil(np.log2((sample_rate / bins_per_octave) * (n_bins / bins_per_octave))))

        # Ensure audio length supports n_fft
        if len(audio) < required_n_fft:
            padding_length = required_n_fft - len(audio)
            audio = np.pad(audio, (0, padding_length), mode='constant')

        # Compute CQT
        cqt = librosa.cqt(audio, sr=sample_rate, n_bins=n_bins, bins_per_octave=bins_per_octave)
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

        # Pad or crop spectrogram to fixed target_shape
        if cqt_db.shape[1] < target_shape[1]:
            pad_width = target_shape[1] - cqt_db.shape[1]
            cqt_db = np.pad(cqt_db, ((0, 0), (0, pad_width)), mode='constant')
        elif cqt_db.shape[1] > target_shape[1]:
            cqt_db = cqt_db[:, :target_shape[1]]

        return cqt_db
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


# Step 6: Data generator
class AudioDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, batch_size, shuffle=True, target_shape=TARGET_SHAPE):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_shape = target_shape
        self.indices = np.arange(len(self.dataframe))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.dataframe) // self.batch_size

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data = self.dataframe.iloc[batch_indices]

        X = []
        y = []

        for _, row in batch_data.iterrows():
            cqt_db = compute_cqt(row['filepath'], target_shape=self.target_shape)
            if cqt_db is not None:
                X.append(cqt_db)
                y.append(row['encoded_label'])

        X = np.array(X)
        X = np.expand_dims(X, axis=-1)  # Add channel dimension for CNN
        y = np.array(y)

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# Step 7: Instantiate data generators
train_generator = AudioDataGenerator(train_df, batch_size=BATCH_SIZE)
valid_generator = AudioDataGenerator(valid_df, batch_size=BATCH_SIZE)
test_generator = AudioDataGenerator(test_df, batch_size=BATCH_SIZE, shuffle=False)

# Step 8: Model definition
def build_model(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    attention = tf.keras.layers.GlobalAveragePooling2D()(x)
    attention = tf.keras.layers.Dense(128, activation='relu')(attention)
    attention = tf.keras.layers.Dense(num_classes, activation='softmax')(attention)

    model = tf.keras.models.Model(inputs, attention)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 9: Build and train the model
input_shape = (*TARGET_SHAPE, 1)  # CQT spectrogram dimensions with channel
model = build_model(input_shape, N_CLASSES)
model.summary()

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS
)

# Step 10: Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Classification report and confusion matrix
y_true = np.concatenate([y for _, y in test_generator], axis=0)
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)


# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
Logger.current_logger().report_matrix("Confusion Matrix", "Test", iteration=EPOCHS, matrix=conf_matrix)

# Save model and encoder
model.save(BASE_DIR.parent / "saved_models/cqt_audio_model.keras")
with open(BASE_DIR.parent / "saved_models/label_encoder.pkl", 'wb') as f:
    pickle.dump(label_encoder, f)

print("Training complete. Model and label encoder saved.")
