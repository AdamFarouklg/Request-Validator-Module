from tensorflow.keras.models import load_model
import tokenize
import pandas as pd
import numpy as np
import string
from string import punctuation
import re
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import io
import tensorflow as tf
from tensorflow import keras
import sklearn.metrics
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt') 
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from keras.layers import Dense, Dropout, Conv1D, MaxPool1D, GlobalMaxPool1D, Embedding, Activation
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


CNN_model = load_model('CNN2.h5')
print(CNN_model)
word2vec_model = Word2Vec.load('word2vec_model2.bin')


w2v_path = '/Users/vu/thesis/word2vec_model2.bin'  # set the path to your pre-trained Word2Vec model
# w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True, unicode_errors='ignore')
CNNCategory = 0
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # text = text.lower()  # Lowercase text
    text = re.sub(f"[{re.escape(punctuation)}]", "", text)  # Remove punctuation
    text = " ".join(text.split())  # Remove extra spaces, tabs, and new lines
    text = re.sub(r"[^A-Za-z0-9\s]+", "", text)
    text = ''.join([word for word in text if word not in string.punctuation])
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def getTokens(input):
    tokensBySlash = str(input.encode('utf-8')).split('/')
    allTokens=[]
    for i in tokensBySlash:
        tokens = str(i).split('-')
        tokensByDot = []
        for j in range(0,len(tokens)):
            tempTokens = str(tokens[j]).split('.')
            tokentsByDot = tokensByDot + tempTokens
        allTokens = allTokens + tokens + tokensByDot
    allTokens = list(set(allTokens))
    if 'com' in allTokens:
        allTokens.remove('com')
    return allTokens

request = input('Enter the request:\n')
CNN_predict = preprocess_text(request)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(CNN_predict)
word_sequences_train = tokenizer.texts_to_sequences(CNN_predict)
vocab_size = len(tokenizer.word_index) + 1
# print(len(tokenizer.word_index))
max_length = 100
CNN_predictor = pad_sequences(word_sequences_train, maxlen=max_length, padding='post')
# sentences = [[tokenizer.index_word[word_index] for word_index in seq] for seq in word_sequences_train]
# # word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=16)

embedding_matrix = np.zeros((vocab_size, 100))
for word, i in tokenizer.word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]

prediction = CNN_model.predict(embedding_matrix)
pred_classes = np.argmax(prediction, axis=1)
pred_class = np.bincount(pred_classes).argmax()

# Load from file
pickle_model = pickle.load(open('LR2.pkl', "rb"))
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

X_predict = [request]
X_predict = vectorizer.transform(X_predict)
regression_predict = pickle_model.predict(X_predict)
# print(type(regression_predict))
logistic_result = np.array2string(regression_predict)
# print(logistic_result)
if "bad" in str(logistic_result) and pred_class != CNNCategory:
    print("The request are malicious. Blocked ")
else:
    print("The request are normal. Allowed ")
# Calculate the accuracy score and predict target values