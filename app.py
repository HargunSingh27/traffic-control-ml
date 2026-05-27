# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import joblib

# ---------------------------------------
# CACHED MODEL + SCALER LOADING
# ---------------------------------------
@st.cache_data
def load_model_and_scaler():
    try:
        model = tf.keras.models.load_model('music_genre_cnn.h5', compile=False)
        scaler = joblib.load('scaler.joblib')

        genre_mapping = {
            0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop',
            5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'
        }

        return model, scaler, genre_mapping

    except FileNotFoundError as e:
        st.error(f"Missing file: {e}. Ensure model & scaler exist in the folder.")
        st.stop()

    except Exception as e:
        st.error(f"Unexpected error loading model/scaler: {e}")
        st.stop()


# ---------------------------------------
# AUDIO FEATURE EXTRACTION
# ---------------------------------------
def extract_features(audio_file, sample_rate=22050, n_mfcc=13, n_chroma=12):
    try:
        y, sr = librosa.load(audio_file, sr=sample_rate, duration=30)

        mfccs_mean = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc), axis=1)
        chroma_mean = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma), axis=1)
        spec_cent_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_roll_mean = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y))

        features = np.concatenate([
            mfccs_mean,
            chroma_mean,
            np.array([spec_cent_mean]),
            np.array([spec_roll_mean]),
            np.array([zcr_mean])
        ])

        return features

    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None


# ---------------------------------------
# PREDICTION PIPELINE
# ---------------------------------------
def predict_genre(audio_file):

    model, scaler, genre_mapping = load_model_and_scaler()

    features = extract_features(audio_file)
    if features is None:
        return "Error: Could not process audio file."

    try:
        features_scaled = scaler.transform(features.reshape(1, -1))
    except Exception as e:
        st.error(f"Error during feature scaling: {e}")
        return "Error: Scaling failed."

    features_cnn = np.expand_dims(features_scaled, axis=-1)

    try:
        prediction_probs = model.predict(features_cnn)
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        return "Error: Prediction failed."

    predicted_index = np.argmax(prediction_probs)
    return genre_mapping.get(predicted_index, "Unknown Genre")


# ---------------------------------------
# STREAMLIT UI
# ---------------------------------------
def main():
    """
    The primary function that builds and runs our Streamlit application.
    """
    st.title("🎵 Music Genre Classification App")
    st.write(
        "Welcome! This application uses a Convolutional Neural Network (CNN) to "
        "predict the genre of a music track."
    )
    st.write(
        "**Instructions:** Please upload a short audio file in `.wav` format to get started."
    )

    # --- File Uploader Widget ---
    uploaded_file = st.file_uploader(
        "Drag and drop your audio file here",
        type=['wav']
    )

    # If a file is uploaded
    if uploaded_file is not None:

        # Audio player so user can listen
        st.audio(uploaded_file, format='audio/wav')

        # Prediction process with spinner
        with st.spinner("Classifying your track... 🎶"):
            # Call your full prediction pipeline
            predicted_genre = predict_genre(uploaded_file)

        # After prediction completes
        st.success("Prediction completed!")

        # Display result
        st.subheader(f"🎼 Predicted Genre: **{predicted_genre}**")
if __name__ == '__main__':
    main()



