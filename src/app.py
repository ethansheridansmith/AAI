import streamlit as st
import librosa as li
import librosa.display
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import keras.backend as K


# NECESARY 
# create a singal spectogram image of the 1 audio file
# set the model as the pre saved model 
# fit the spectogram to the model 
# display results as bar chart

# OPTIONAL
# in info page have the loss values and other groahs of the cnn model when we train it

@st.cache(allow_output_mutation=True)
def load_model(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={'f1_score': f1_score})

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

model_path = "genre_model.h5"
model = load_model(model_path)

def generate_spectrogram(audio_file):
    y, sr = li.load(audio_file, sr= None, mono=True)
    S = li.feature.melspectrogram(y=y, sr=sr)
    S_dB = li.power_to_db(S, ref=np.max)
    return S_dB

def display_spectrogram(S_dB):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    st.pyplot()
    
def classify_spectrogram(S_dB):
    img = Image.fromarray(S_dB).convert('RGBA')
    img = img.resize((432, 288))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    prediction = model.predict(img_array)
    return prediction


st.title("Audio Classification App")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac", "ogg"])

if uploaded_file is not None:
    with st.spinner('Generating spectrogram...'):
        S_dB = generate_spectrogram(uploaded_file)
    st.success('Spectrogram generated.')
    
    display_spectrogram(S_dB)

    with st.spinner('Classifying audio...'):
        prediction = classify_spectrogram(S_dB)
    st.success('Audio classified.')

    st.bar_chart(prediction)

    # If you want to display model training information, add an "info" page
    if st.button("Show Model Training Info"):
        st.subheader("Loss values and other graphs of the CNN model during training")
        # Display your loss values and graphs here using st.pyplot() or st.write()