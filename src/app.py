import math
import tempfile

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pydub
import streamlit as st
from keras.initializers import glorot_uniform
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Model
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mutagen.mp3 import MP3
from pydub import AudioSegment
from tensorflow.keras.preprocessing.image import img_to_array, load_img

pydub.AudioSegment.converter = r"C:\FFmpeg\bin\ffmpeg.exe"


###========================================================###
###                   USEFUL FUNCTIONS                     ###
###========================================================###


def create_model(input_shape, classes):
    def create_step_in_model(dim, input_tensor):
        X = Conv2D(dim, kernel_size=(3, 3), strides=(1, 1))(input_tensor)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        return MaxPooling2D((2, 2))(X)

    X_input = Input(input_shape)
    X = X_input

    layer_dimensions = [8, 16, 32, 64, 128, 256]

    for layer_dim in layer_dimensions:
        X = create_step_in_model(layer_dim, X)

    X = Flatten()(X)
    X = Dropout(rate=0.3)(X)
    X = Dense(classes, activation='softmax', name=f'fc{classes}', kernel_initializer=glorot_uniform(seed=9))(X)
    model = Model(inputs=X_input, outputs=X, name='cnn')
    return model


# Predict the genres.
def predict(image_data, model):
    image = img_to_array(image_data).reshape((1, 288, 432, 4))
    prediction = model.predict(image / 255)
    prediction = prediction.reshape((10,))
    class_label = np.argmax(prediction)
    return class_label, prediction


# Create our wav file for conversion to spectrogram.
def create_wav(file, start, duration):
    wav = AudioSegment.from_file(file)
    start_position = 1000 * start
    wav = wav[start_position:start_position + (1000 * duration)]
    wav.export("extracted.wav", format='wav')


# Create our spectrogram from our wav file.
def create_spectrogram(wav_file, check_period):
    y, sr = librosa.load(wav_file, duration=check_period)
    mels = librosa.feature.melspectrogram(y=y, sr=sr)

    fig = plt.Figure()
    FigureCanvasAgg(fig)
    plt.imshow(librosa.power_to_db(mels, ref=np.max))

    # Save the spectrogram locally for insertion into model.
    plt.savefig('final-spectrogram.png')


@st.cache_data
def process_file(uploaded_file):
    with st.spinner('Processing audio file...'):
        # ... your processing code here
        # Prediction data collection.
        predictions = []
        outcome = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        progress_bar = st.progress(0)

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            test_file = temp_file.name
            audio_file = MP3(test_file)

        total_chunks = math.floor(audio_file.info.length / periods)
        for i in range(1, total_chunks):
            create_wav(test_file, i * 10, check_period)
            create_spectrogram("extracted.wav", check_period)
            image_data = load_img('final-spectrogram.png', color_mode='rgba', target_size=(288, 432))
            class_label, prediction = predict(image_data, model)
            prediction = prediction.reshape((10,))
            outcome = outcome + prediction
            predictions.append(prediction)  # model.predict returns a 2D array, we just want the first row

            progress = i / (total_chunks - 1)
            progress_bar.progress(progress)

        return predictions, outcome


###========================================================###
###                      MAIN HANDLING                     ###
###========================================================###


genre_labels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
check_period = 3
periods = 15

model = create_model(input_shape=(288, 432, 4), classes=len(genre_labels))
model.load_weights("genre_model_98.h5")

st.title("Song Genre Classification")

uploaded_file = st.file_uploader("Choose an audio file", type=["mp3"])

if uploaded_file is not None:
    predictions, outcome = process_file(uploaded_file)

    st.success('Audio file processed.')

    class_label = np.argmax(outcome)
    st.write(f"### Genre Prediction: {genre_labels[class_label]}")

    # Display the overall predicted genres.

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x=genre_labels, height=outcome,
           color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf'])
    plt.xticks(rotation=40)
    ax.set_title("Probability composition of Genres")
    st.pyplot(fig)

    # Print the rolling prediction

    plt.clf()
    predictions = np.array(predictions)
    n_segments = predictions.shape[0]
    n_genres = predictions.shape[1]
    time = np.arange(n_segments) * periods

    selected_genres = st.multiselect('Select genres', genre_labels, default=genre_labels)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i in range(n_genres):
        if genre_labels[i] in selected_genres:
            ax.plot(time, predictions[:, i], label=genre_labels[i])

    plt.xlabel('Time (s)')
    plt.title('Genre Predictions Over Time')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
    ax.set_title("Probability composition of Genres")
    st.pyplot(fig)

    # Display the spectrogram.

    st.write(f"### Mel Spectrogram")
    st.image("final-spectrogram.png", use_column_width=True)

# If you want to display model training information, add an "info" page
if st.button("Show Model Training Info"):
    st.subheader("Loss values and other graphs of the CNN model during training")
    # Display your loss values and graphs here using st.pyplot() or st.write()
