import librosa
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Define the extract_features function
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

# Load the trained model from the pickle file
with open("D:\\Coding\\python\\Hackelite\\trained_model.pkl", 'rb') as file:
    clf = pickle.load(file)

# Load the scaler object used during training
with open("D:\\Coding\\python\\Hackelite\\scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)

def predict_emotion(file_path):
    try:
        # Extract features from the new audio file
        features = extract_features(file_path)
        
        if features is not None:
            # Scale the features using the same scaler used during training
            features_scaled = scaler.transform([features])

            # Use the trained model to predict the emotion
            emotion_prediction = clf.predict(features_scaled)[0]
            return emotion_prediction
        else:
            return None
    except Exception as e:
        print(f"Error predicting emotion for {file_path}: {e}")
        return None

# Example usage:
new_audio_file_path = "D:\\Coding\\python\\Hackelite\\audio.wav"
predicted_emotion = predict_emotion(new_audio_file_path)

if predicted_emotion is not None:
    print(f"Predicted Emotion: {predicted_emotion}")
else:
    print("Emotion prediction failed.")
