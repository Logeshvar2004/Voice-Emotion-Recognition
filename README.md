# Voice-Emotion-Recognition
This project implements a speech emotion recognition system using machine learning. It includes feature extraction from audio files, model training using the MLPClassifier, and emotion prediction for new audio samples.

## Dataset : https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
The RAVDESS dataset, or Ryerson Audio-Visual Database of Emotional Speech and Song, is a collection of emotional speech and song recordings. It consists of 7356 files, featuring 24 professional actors who simulate various emotions, including neutral, calm, happy, sad, angry, fearful, disgust, and surprised. Each audio file is labeled with the corresponding emotion, providing a diverse and rich resource for training and evaluating speech emotion recognition models.

## Features
- Extracts features like MFCCs, Chroma, Mel Spectrogram, Spectral Contrast, and Tonnetz.
- Trains a Multilayer Perceptron (MLP) Classifier for emotion recognition.
- Scales features using StandardScaler for improved model performance.

## Files:
1) trained_model.pkl: This file contains the serialized form of your trained machine learning model, specifically an MLPClassifier in this case. The model is saved using the pickle module, allowing you to persist the trained model to disk. Later, you can load this file to make predictions without retraining the model.

2) scaler.pkl: The scaler object used during training is saved in this file. The scaler is crucial for preprocessing input data in a consistent manner. It contains the mean and standard deviation values learned from the training set, ensuring that any new data is scaled in the same way as the training data. This helps maintain consistency between the training and prediction phases.

3) test.py: This file is used for testing purpose.

4) streamlit.py: This machine learning model and related functionality are now accessible through a web interface. Users can interact with your model by uploading audio files directly on the Streamlit app. simply it's a deployment process.

In conclusion, this project utilizes the RAVDESS dataset for emotion classification based on audio features. The trained model (saved in 'trained_model.pkl') and the scaler (saved in 'scaler.pkl') are crucial components for predicting emotions from new audio files. By deploying the project on Streamlit, I've transformed it into an accessible web application, allowing users to conveniently predict emotions in audio files through a user-friendly interface. This deployment enhances the project's usability, making it available to a broader audience without the need for local code execution. The combination of the RAVDESS dataset, machine learning model, and web deployment through Streamlit represents a comprehensive and accessible solution for emotion prediction in audio files.
