import os
import librosa
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer


def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate), axis=1)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate), axis=1)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate), axis=1)
        tonnetz = np.mean(librosa.feature.tonnetz(y=audio, sr=sample_rate), axis=1)

        return np.hstack((mfccs, chroma, mel, spectral_contrast, tonnetz))
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

data_directory = "D:\Coding\python\Hackelite\Audio"
emotion_labels = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
X, y = [], []

for root, dirs, files in os.walk(data_directory):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            try:
                emotion_label = emotion_labels[file.split('-')[2]]
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(emotion_label)
            except KeyError:
                print(f"Emotion label not found for file: {file}")

X = np.array(X)
y = np.array(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42, alpha=0.01, solver='adam', batch_size=32, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
cv_accuracy = cross_val_score(clf, X_scaled, y, cv=5, scoring=make_scorer(accuracy_score))
print(cv_accuracy)

with open('trained_model.pkl', 'wb') as file:
    pickle.dump(clf, file)