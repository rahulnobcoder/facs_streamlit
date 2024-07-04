import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import cv2
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import pandas as pd
from tempfile import NamedTemporaryFile
from functions import *
au_to_movements= {
    'au1': 'inner brow raiser',
    'au2': 'outer brow raiser',
    'au4': 'brow lowerer',
    'au5': 'upper lid raiser',
    'au6': 'cheek raiser',
    'au9': 'nose wrinkler',
    'au12': 'lip corner puller',
    'au15': 'lip corner depressor',
    'au17': 'chin raiser',
    'au20': 'lip stretcher',
    'au25': 'lips part',
    'au26': 'jaw drop'
}
au_labels = [
    "au1",
    "au12",
    "au15",
    "au17",
    "au2",
    "au20",
    "au25",
    "au26",
    "au4",
    "au5",
    "au6",
    "au9"
]
col=[au_to_movements[i] for i in au_labels]
def binary_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss(y_true, y_pred):
        # Define epsilon to avoid log(0)
        epsilon = tf.keras.backend.epsilon()
        # Clip predictions to prevent log(0) and log(1 - 0)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        # Compute the focal loss
        fl = - alpha * (y_true * (1 - y_pred)**gamma * tf.math.log(y_pred)
                       + (1 - y_true) * (y_pred**gamma) * tf.math.log(1 - y_pred))
        return tf.reduce_mean(fl, axis=-1)
    return focal_loss

loss = binary_focal_loss(gamma=2.0, alpha=0.25)

# Function to read video frames into a list
def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

# Function to process frames and make predictions
def process_frames(frames, model):
    frames = [get_face(frame) for frame in tqdm(frames[:len(frames)-1])]
    st.text(f"face shape : {frames[0].shape}")
    frame_array = np.array(frames)
    preds = model.predict(frame_array).round()
    return preds

# Function to save predictions to a CSV file
def save_predictions_to_csv(predictions, filename="predictions.csv"):
    df = pd.DataFrame(predictions,columns=col)
    df.to_csv(filename, index=False)
    return filename

# Load your Keras model
def load_model():
    model = tf.keras.models.load_model('incept_v3_10fps_full_dp0.2.keras',
                                       custom_objects={'binary_focal_loss': binary_focal_loss})
    return model

# Streamlit app
def main():
    st.title("Video Frame Prediction App")
    
    # Upload video file
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        with NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        # Load the model
        model = load_model()

        # Predict button
        if st.button("Predict"):
            # Read frames from video
            st.text("Reading video frames...")
            frames = read_video_frames(video_path)
            st.text(f"Total frames read: {len(frames)}")

            # Process frames and make predictions
            st.text("Processing frames and making predictions...")
            predictions = process_frames(frames, model)
            st.text("Predictions completed!")

            # Save predictions to CSV
            csv_file_path = save_predictions_to_csv(predictions)
            st.text("Predictions saved to CSV!")

            # Make CSV downloadable
            with open(csv_file_path, "rb") as f:
                st.download_button(
                    label="Download CSV",
                    data=f,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

            # Clean up the temporary file
            os.remove(video_path)

if __name__ == "__main__":
    main()
