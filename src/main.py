import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.preprocessing.sequence import pad_sequences

train_df = pd.read_csv('train.csv')
# Selecting Edgar Allen Poe as author style to emulate
author = train_df[train_df['author'] == 'EAP']["text"]
print('Number of training sentences: ',author.shape[0])

max_words = 50000 # Max size of the dictionary
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(author.values)
sequences = tokenizer.texts_to_sequences(author.values)

# Flatten the list of lists resulting from the tokenization. This will reduce the list
# to one dimension, allowing us to apply the sliding window technique to predict the next word
text = [item for sublist in sequences for item in sublist]
vocab_size = len(tokenizer.word_index)

sentence_len = 20
pred_len = 1
train_len = sentence_len - pred_len
seq = []
# Sliding window to generate train data
for i in range(len(text)-sentence_len):
    seq.append(text[i:i+sentence_len])
# Reverse dictionary to decode tokenized sequences back to words
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

# Each row in seq is a 20 word long window. We append he first 19 words as the input to predict the 20th word
trainX = []
trainy = []
for i in seq:
    trainX.append(i[:train_len])
    trainy.append(i[-1])

# define model
model_2 = Sequential([
    Embedding(vocab_size+1, 50, input_length=train_len),
    LSTM(100, return_sequences=True),
    LSTM(100),
    Dense(100, activation='relu'),
    Dropout(0.1),
    Dense(vocab_size, activation='softmax')
])
model_2.add( keras.layers.Flatten())

# Train model with checkpoints
model_2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
filepath = "./model_2_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
history = model_2.fit(np.asarray(trainX),
         np.asarray(trainy),
         epochs = 300,
         batch_size = 128,
         callbacks = callbacks_list,
         verbose = 1)

def gen(model,seq,max_len = 20):
    ''' Generates a sequence given a string seq using specified model until the total sequence length
    reaches max_len'''
    # Tokenize the input string
    tokenized_sent = tokenizer.texts_to_sequences([seq])
    max_len = max_len+len(tokenized_sent[0])
    # If sentence is not as long as the desired sentence length, we need to 'pad sequence' so that
    # the array input shape is correct going into our LSTM. the `pad_sequences` function adds 
    # zeroes to the left side of our sequence until it becomes 19 long, the number of input features.
    while len(tokenized_sent[0]) < max_len:
        padded_sentence = pad_sequences(tokenized_sent[-19:],maxlen=19)
        op = model.predict(np.asarray(padded_sentence).reshape(1,-1))
        tokenized_sent[0].append(op.argmax()+1)
        
    return " ".join(map(lambda x : reverse_word_map[x],tokenized_sent[0]))

print()
print(gen(model_2, 'In his left hand', 20))
