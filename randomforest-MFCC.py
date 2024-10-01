"""
using the RandomForestClassifier from Scikit-learn as the model for classifying bird species based on their MFCC features extracted from audio files.
======================with noise cancalation and cut file =========================
Test Accuracy: 0.71
                   precision    recall  f1-score   support

Black-headed Gull       0.00      0.00      0.00         1
      Brent Goose       0.00      0.00      0.00         1
        Chaffinch       0.00      0.00      0.00         1
     Common Crane       0.00      0.00      0.00         2
 Common Crossbill       0.00      0.00      0.00         1
             Coot       0.00      0.00      0.00        18
        Eagle Owl       0.00      0.00      0.00         2
        Fieldfare       0.00      0.00      0.00         1
        Goldeneye       0.00      0.00      0.00         1
       Greenfinch       0.00      0.00      0.00         1
     Grey Wagtail       0.00      0.00      0.00         1
          Moorhen       0.00      0.00      0.00         6
        Mute Swan       0.00      0.00      0.00         1
          Redwing       0.74      0.68      0.71       238
      Song Thrush       0.71      0.93      0.81       412
        Tawny Owl       0.69      0.92      0.79        39
       Tree Pipit       0.00      0.00      0.00         4
       Water Rail       0.00      0.00      0.00         1
           Wigeon       0.00      0.00      0.00         1
     unidentified       0.35      0.07      0.12        96

         accuracy                           0.71       828
        macro avg       0.12      0.13      0.12       828
     weighted avg       0.64      0.71      0.66       828
========================without noise cancelation and cut file================
Test Accuracy: 0.53


                 precision    recall  f1-score   support

        Brent Goose       0.00      0.00      0.00         1
          Bullfinch       0.00      0.00      0.00         2
          Chaffinch       0.00      0.00      0.00         1
       Common Crane       0.00      0.00      0.00         1
   Common Sandpiper       0.00      0.00      0.00         1
               Coot       0.00      0.00      0.00        11
          Eagle Owl       0.00      0.00      0.00         5
    Green Sandpiper       0.00      0.00      0.00         1
         Greenfinch       0.00      0.00      0.00         2
            Mallard       0.00      0.00      0.00         1
            Moorhen       0.00      0.00      0.00        12
            Redwing       0.39      0.26      0.31       246
        Song Thrush       0.56      0.87      0.68       410
          Tawny Owl       0.67      0.11      0.19        37
         Tree Pipit       0.00      0.00      0.00         4
         Water Rail       0.00      0.00      0.00         2
White-fronted Goose       0.00      0.00      0.00         1
             Wigeon       0.00      0.00      0.00         1
       unidentified       0.52      0.15      0.23        89

           accuracy                           0.53       828
          macro avg       0.11      0.07      0.07       828
       weighted avg       0.48      0.53      0.46       828
"""
import numpy as np
import librosa
import os
import re
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def read_lookup_file(file_path):
    lookup_dict = {}
    with open(file_path, 'r') as file:
        content = file.read()
        matches = re.findall(r"\'(.*?)\',\s*\'(.*?)\'", content)
        for match in matches:
            bird_id, bird_name = match
            lookup_dict[bird_id] = bird_name
    return lookup_dict


def extract_bird_identifier(filename):
    parts = filename.split('___')[-1]
    bird_id = parts.split('.')[0]
    return bird_id


def extract_mfcc(file_path, max_pad_len=100):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

        if len(audio) == 0:
            raise ValueError(f"Empty audio file: {file_path}")

        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]

        return mfcc

    except Exception as e:
        warnings.warn(f"Error processing file {file_path}: {str(e)}")
        return None


def load_dataset(data_folder, lookup_dict):
    labels = []
    mfccs = []

    for file_name in os.listdir(data_folder):
        if file_name.endswith('.wav'):
            file_path = os.path.join(data_folder, file_name)

            mfcc = extract_mfcc(file_path)

            if mfcc is not None:
                bird_id = extract_bird_identifier(file_name)
                bird_label = lookup_dict.get(bird_id, 'unidentified')
                mfccs.append(mfcc)
                labels.append(bird_label)

    return np.array(mfccs), np.array(labels)


# Preprocess the data
data_folder = r'C:\Users\kian_\PycharmProjects\IoT_System\Data'
lookup_file = r'c:\bird\lookup.txt'
lookup_dict = read_lookup_file(lookup_file)

X, y = load_dataset(data_folder, lookup_dict)
X = X.reshape(X.shape[0], -1)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

test_accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

y_pred = model.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

unique_test_classes = np.unique(y_test)
filtered_class_names = encoder.inverse_transform(unique_test_classes)

print(classification_report(y_test, y_pred, target_names=filtered_class_names))


def extract_mfcc_from_new_file(file_path, max_pad_len=100):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

        if len(audio) == 0:
            raise ValueError(f"Empty audio file: {file_path}")

        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]

        return mfcc

    except Exception as e:
        warnings.warn(f"Error processing file {file_path}: {str(e)}")
        return None


def predict_bird_species(file_path):
    mfcc = extract_mfcc_from_new_file(file_path)

    if mfcc is not None:
        mfcc = mfcc.reshape(1, -1)
        predicted = model.predict(mfcc)

        predicted_class_label = encoder.inverse_transform([predicted])[0]

        print(f"Predicted bird species: {predicted_class_label}")
    else:
        print(f"Unable to process the file: {file_path}")


new_bird_sound_path = r'c:birdtest\1.wav'
predict_bird_species(new_bird_sound_path)
