"""
NAME: imdb.py
AUTHOR: MTHANDAZO NDHLOVU
DESCRIPTION: BASIC NEURAL NETWORK WHICH ILLUSTRATES BINARY CLASSIFICATION AND THE USAGE
OF BINARY_CROSSENTROPY
DATASET: IMDB (INTERNET MOVIE DATABASE)
ACKNOWLEDGEMENTS: DEEP LEARNING WITH PYTHON, KERAS

"""

#imports
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras import layers

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

"""
decoding integer sequences back to sentenses
word_index = imdb.get_word_index()
reverse the word_index which is simple a dictionary mapping words to integers
reverse_word_index = dict([(value, key) for (key, value) in word_index.times()])
decode the review, note that indices where offset by 3 because 0, 1 and 2 are
reserved indices for padding, start of sequence and unknown
decode_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

"""

def vectorize_sequences(sequences, dimension=10000):
    #create an all zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequences in enumerate(sequences):
        results[i, sequences] = 1 #set specific indices of results[i] to 1s
    return results
#our vector training data
x_train = vectorize_sequences(train_data)
#our vectorized test data
x_test = vectorize_sequences(test_data)

#our vectorized labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#create the network
network = Sequential()
network.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
network.add(layers.Dense(16, activation='relu'))
network.add(layers.Dense(1, activation='sigmoid'))

network.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])
#validation of our model
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = network.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val,y_val))
history_dict = history.history
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show
