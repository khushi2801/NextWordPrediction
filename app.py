from unittest import result
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import heapq

SEQUENCE_LENGTH = 40
chars_size = 71
char_indices = pickle.load(open('char_indices.pkl', 'rb'))
indices_char = pickle.load(open('indices_char.pkl', 'rb'))
chars = np.load('chars.npy')

app = Flask(__name__)

# Load model
model = load_model('model.h5')

@app.route('/', methods = ['GET','POST'])
def home():
    result = ' '
    return render_template('index.html', result = str(result))

@app.route('/predict', methods = ['GET','POST'])
def predict():
    text = ''
    if request.method == 'POST':
        text = request.form['text']
        seq = text[:SEQUENCE_LENGTH].lower()
        result = str(predict_completions(seq, 3))
        return render_template('index.html', result = str(result))

# Predict completions function: This function wraps everything and allows us to predict multiple completions
def predict_completions(text, n = 3):
    x = prepare_input(text)
    pred = model.predict(x, verbose = 0)[0]
    next_indices = sample(pred, n)
    return  [indices_char[i] + predict_completion(text[1:] + indices_char[i]) for i in next_indices]

# Prepare input function: This function creates input features for given sentence
def prepare_input(text):
    x = np.zeros((1, SEQUENCE_LENGTH, chars_size))
    for t, char in enumerate(text):
        x[0, t, char_indices[char]] = 1.0
    return x

# Sample function: This function allows us to ask our model what are the next probable characters
def sample(pred, top_n = 3):
    pred = np.asarray(pred).astype('float64')
    pred = np.log(pred)
    exp_pred = np.exp(pred)
    pred = exp_pred / np.sum(exp_pred)
    return heapq.nlargest(top_n, range(len(pred)), pred.take)

# Predict completion function: This function is used to predict next characters
def predict_completion(text):
    original_text = text
    completion = ''
    while True:
        x = prepare_input(text)
        pred = model.predict(x, verbose=0)[0]
        next_index = sample(pred, top_n=1)[0]
        next_char = indices_char[next_index]
        text = text[1:] + next_char
        completion += next_char
        if len(original_text + completion) + 2 > len(original_text) and next_char == ' ':
            return completion

if __name__ == '__main__':
    app.run(debug = True)