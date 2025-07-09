from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import os

# Ensure required NLTK data is available
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

app = Flask(__name__)

# Load trained model
model = load_model('model.h5')

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load max_length dynamically from file
with open('max_length.pkl', 'rb') as f:
    max_length = pickle.load(f)

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form['tweet']
    processed_tweet = preprocess_text(tweet)
    seq = tokenizer.texts_to_sequences([processed_tweet])
    padded = pad_sequences(seq, maxlen=max_length, padding='post')

    prediction = model.predict(padded)[0][0]
    label = int(prediction > 0.5)
    confidence = round(prediction * 100, 2)

    if label == 1:
        result = "The tweet indicates suicide ideation."
    else:
        result = "The tweet does not indicate suicide ideation."

    return render_template('index.html', tweet=tweet, prediction=result, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
