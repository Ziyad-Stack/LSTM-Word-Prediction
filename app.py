import pickle

import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

#Load the model

model = load_model('next_word_lstm.h5')

#Load the Tokenizer

with open('tokenizer.pickle','rb') as handle:
    Tokenizer= pickle.load(handle)


##Function to predict the next word:

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_next_word(model, text, tokenizer, max_sequence_len):
    # 1. Convert text to a clean list of numbers inside a 2D matrix structure
    token_list = tokenizer.texts_to_sequences([text])
    
    # 2. Extract the first inner list directly to look at raw word count safely
    raw_tokens = token_list[0]
    
    # 3. Truncate tokens if the text length exceeds your model parameters
    if len(raw_tokens) >= max_sequence_len:
        raw_tokens = raw_tokens[-(max_sequence_len-1):]
        
    # 4. Pass the 1D list inside a single array wrapper to output a clean 2D matrix shape
    padded_input = pad_sequences([raw_tokens], maxlen=max_sequence_len-1, padding='pre')
    
    # 5. Get the raw prediction array from your model layers
    predicted = model.predict(padded_input, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)

    # 6. Map the resulting numeric array ID back to a readable text string
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
            
    return None

#streamlit app

st.title("Next Word Prediction with LSTM ")
input_text = st.text_input("Enter the sequence of Word")
if st.button("Predict the next word"):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, input_text, Tokenizer, max_sequence_len)
    st.write(f"Next Word Prediction: {next_word}")