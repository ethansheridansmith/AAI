# import streamlit as st
# import librosa as li
# import librosa.display
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import keras.backend as K
# from keras.preprocessing.image import img_to_array, load_img
# import pydub
# from pydub import AudioSegment
# from matplotlib.backends.backend_agg import FigureCanvasAgg
# import tempfile

# #####################
# ### /\ imports /\ ###
# #####################

# ##################
# ### \/ code \/ ###
# ##################

# # used for extracting segmetns of songs for spectogram analysis and the pydub library
# pydub.AudioSegment.converter = r"C:\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"

# #############################################
# # model load function, uses f1 score as
# # custom object
# #############################################

# @st.cache(allow_output_mutation=True)
# def load_model(model_path):
#     return tf.keras.models.load_model(model_path, custom_objects={'f1_score': f1_score})

# #############################################
# # score function for evaluation of the model   
# #############################################

# def f1_score(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     recall = true_positives / (possible_positives + K.epsilon())
#     f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
#     return f1_val

# #############################################
# # model is loaded, trained in previous 
# # program 
# #############################################

# model_path = "genre_model_98.h5"
# model = load_model(model_path)

# ############################################
# # genres defined and audio duration defined    
# ############################################
# # Duration for audio extraction
# check_period = 3 
# # genre labels for classification 
# genre_labels = ["blues", "classical", "country", "disco", "hihop",
#                 "jazz", "metal", "pop", "reggea", "rock"]

# ###############################################
# # gets the image and reshapes it 
# ###############################################

# # Predict the genres.
# def predict(image_data, model):
#     image = img_to_array(image_data).reshape((1, 288, 432, 4))
#     prediction = model.predict(image / 255)
#     prediction = prediction.reshape((10, ))
#     class_label = np.argmax(prediction)
#     return class_label, prediction

# # Create our wav file for conversion to spectrogram.
# def create_wav(file_path, start, duration):
#     wav = pydub.AudioSegment.from_file_using_temporary_files(file_path)
#     start_position = 1000 * start
#     wav = wav[start_position:start_position + (1000 * duration)]
#     wav.export("extracted.wav", format='wav')

# # Create our spectrogram from our wav file.
# def create_spectrogram(wav_file):
#     y, sr = librosa.load(wav_file, duration=check_period)
#     mels = librosa.feature.melspectrogram(y=y, sr=sr)
    
#     fig = plt.Figure()
#     FigureCanvasAgg(fig)
#     plt.imshow(librosa.power_to_db(mels, ref=np.max))
    
#     # Save the spectrogram locally for insertion into model.
#     plt.savefig('final-spectrogram.png')

# st.title("Audio Classification App")

# uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac", "ogg"])

# if uploaded_file is not None:
#     with st.spinner('Processing audio file...'):
#         # Save the uploaded file to a temporary file
#         with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#             temp_file.write(uploaded_file.read())
#             temp_file_path = temp_file.name

#         create_wav(temp_file_path, 10, check_period)
#         create_spectrogram("extracted.wav")
#         image_data = load_img('final-spectrogram.png', color_mode='rgba', target_size=(288, 432))

#     st.success('Audio file processed.')

#     class_label, prediction = predict(image_data, model)
#     prediction = prediction.reshape((10,))

#     st.write(f"### Genre Prediction: {genre_labels[class_label]}")
#     fig, ax = plt.subplots(figsize=(6, 4.5))
#     ax.bar(x=genre_labels, height=prediction)
#     plt.xticks(rotation=40)
#     ax.set_title("Probability composition of Genres")
#     st.pyplot(fig)
#     st.write(f"### Mel Spectrogram")
#     st.image("final-spectrogram.png", use_column_width=True)
#     plt.show()

# # If you want to display model training information, add an "info" page
# if st.button("Show Model Training Info"):
#     st.subheader("Loss values and other graphs of the CNN model during training")
#     # Display your loss values and graphs here using st.pyplot() or st.write()

import streamlit as st
import librosa as li
import librosa.display
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import keras.backend as K
from keras.preprocessing.image import img_to_array, load_img
import pydub
from pydub import AudioSegment
from matplotlib.backends.backend_agg import FigureCanvasAgg
import tempfile
import math
from mutagen.mp3 import MP3

# Manually set ffmpeg path to fix errors.
pydub.AudioSegment.converter = r"C:\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"

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

model_path = "genre_model_98.h5"
model = load_model(model_path)

check_period = 3
genre_labels = ["blues", "classical", "country", "disco", "hihop",
                "jazz", "metal", "pop", "reggea", "rock"]

def predict(image_data, model):
    image = img_to_array(image_data).reshape((1, 288, 432, 4))
    prediction = model.predict(image / 255)
    prediction = prediction.reshape((10, ))
    class_label = np.argmax(prediction)
    return class_label, prediction

def create_wav(file_path, start, duration):
    wav = pydub.AudioSegment.from_file_using_temporary_files(file_path)
    start_position = 1000 * start
    wav = wav[start_position:start_position + (1000 * duration)]
    wav.export("extracted.wav", format='wav')

def create_spectrogram(wav_file):
    y, sr = librosa.load(wav_file, duration=check_period)
    mels = librosa.feature.melspectrogram(y=y, sr=sr)
    
    fig = plt.Figure()
    FigureCanvasAgg(fig)
    plt.imshow(librosa.power_to_db(mels, ref=np.max))
    
    plt.savefig('final-spectrogram.png')

def over_time_graph(predictions):
    predictions = np.array(predictions)
    n_segments = predictions.shape[0]
    n_genres = predictions.shape[1]
    time = np.arange(n_segments) * periods

    plt.figure()

    for i in range(n_genres):
        plt.plot(time, predictions[:, i], label=genre_labels[i])

    plt.xlabel('Time (s)')
    plt.ylabel('Prediction Probability')
    plt.title('Genre Predictions Over Time')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
    plt.show()

def bar_chart_predictions(outcome):
    plt.figure()
    plt.bar(genre_labels, outcome)
    plt.xlabel('Genre')
    plt.ylabel('Prediction Probability')
    plt.title('Genre Predictions')
    plt.xticks(rotation=40)
    plt.show()

st.title("Audio Classification App")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac", "ogg"])

# audio = MP3(uploaded_file)

if uploaded_file is not None:
    with st.spinner('Processing audio file...'):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        create_wav(temp_file_path, 10, check_period)
        create_spectrogram("extracted.wav")
        image_data = load_img('final-spectrogram.png', color_mode='rgba', target_size=(288, 432))

    st.success('Audio file processed.')

    class_label, prediction = predict(image_data, model)
    prediction = prediction.reshape((10,))

    st.write(f"### Genre Prediction: {genre_labels[class_label]}")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x=genre_labels, height=prediction)
    plt.xticks(rotation=40)
    ax.set_title("Probability composition of Genres")
    st.pyplot(fig)
    st.write(f"### Mel Spectrogram")
    st.image("final-spectrogram.png", use_column_width=True)
    plt.show()

if st.button("Show Genre Predictions Over Time"):
    st.subheader("Genre Predictions Over Time")
    predictions = []
    outcome = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    periods = 15
    total_chunks = math.floor(audio.info.length / periods)

    for i in range(1, total_chunks):
        create_wav(temp_file_path, i * 10, check_period)
        create_spectrogram("extracted.wav")
        image_data = load_img('final-spectrogram.png', color_mode='rgba', target_size=(288, 432))
        class_label, prediction = predict(image_data, model)
        prediction = prediction.reshape((10,))
        outcome = outcome + prediction
        predictions.append(prediction)

    over_time_graph(predictions)
    st.pyplot()

    plt.clf()

    bar_chart_predictions(outcome)
    st.pyplot()

    plt.clf()
