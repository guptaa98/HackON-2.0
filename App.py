
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import text_hammer as th

# Keras
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense , Bidirectional
from keras.layers.embeddings import Embedding
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
#from keras.preprocessing.text import Tokenizer
from keras.models import load_model

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'Models/Emotion_Detection.h5'

# Load your trained model
model = load_model(MODEL_PATH)
     
#print('Model loaded. Check http://127.0.0.1:5000/')


@app.route('/', methods=['GET','POST'])
def index():
    # Main page
    return render_template('Home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        
        def get_key(value):
            dictionary={'joy':0,'anger':1,'love':2,'sadness':3,'fear':4,'surprise':5}
            for key,val in dictionary.items():
                if (val==value):
                    return key
        
        num_words = 5000 
        tokenizer = Tokenizer( num_words , lower=True )

        message = request.form['message']

        sentence_lst = []
        sentence_lst.append(message)
        sentence_seq = tokenizer.texts_to_sequences(sentence_lst)
        sentence_padded = pad_sequences(sentence_seq,maxlen=300,padding='post')
        my_prediction = get_key(model.predict_classes(sentence_padded))
        
        #my_prediction = model.predict(data_pad)
        print (my_prediction)
        return render_template('result.html', prediction = my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
