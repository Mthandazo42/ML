#imports
from keras.datasets import reuters
from keras.models import Sequential
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labes), dimension))
    for i, label in enumerate(labels):
        results[i, labels] = 1
    return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

"""
ALTERNATIVE WAY OF ENCODING THE LABELS THE KERAS WAY
from keras.utils.np_utils import to_categorical

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

"""

model = Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

def get_categorical_accuracy_keras(x_test, y_train):
    return K.mean(K.equal(K.argmax(x_test, axis=1), K.argmax(y_train, axis=1)))

model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=[get_categorical_accuracy_keras])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

"""
LIKE THE IMDB PROGRAM WE CAN START WITH 20 EPOCHS AND OBSERVE IF THERE IS OVERFITTING OR NOT
AND REDUCE THE NUMBER OF EPOCHS TO ELIMINATE OVERFITTING

"""

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=9,
                    batch_size=512,
                    validation_data=(x_val,y_val))
"""

"""

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
