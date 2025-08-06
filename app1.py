# import necessary libraries
from flask import Flask, render_template, request, jsonify
import numpy as np
import librosa
import librosa.display
from keras.models import load_model

app = Flask(__name__)

# Load your trained model
model = load_model('Live_Event_model.h5')

# Define an endpoint for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define an endpoint for the home page
@app.route('/home')
def home():
    return render_template('home.html')

# Define an endpoint for the login page
@app.route('/login')
def login():
    return render_template('login.html')

# Define an endpoint to handle audio file submission
@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'})

    audio_file = request.files['audio']

    if audio_file.filename == '':
        return jsonify({'error': 'No selected audio file'})

    try:
        # Load and process the audio file
        y, sr = librosa.load(audio_file, duration=3, offset=0.5)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        mfcc = np.expand_dims(mfcc, axis=0)
        mfcc = np.expand_dims(mfcc, axis=2)

        # Make a prediction using the model
        prediction = model.predict(mfcc)
        predicted_label = np.argmax(prediction)
        emotions = ['children_playing', 'Dog_Bark', 'Drilling', 'Gunshot','Scream']
        predicted_emotion = emotions[predicted_label]

        return jsonify({'emotion': predicted_emotion})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
