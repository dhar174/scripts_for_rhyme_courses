import itertools
import pandas as pd
import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import random
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
import io

from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

import collections
import pathlib
import re
import string


from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras import utils
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import tensorflow_datasets as tfds
import tensorflow_text as tf_text

from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

import os

vocab_size = 10000

dtype = {
    "text": 'str',
    "label": 'uint8',
}


seed=42

raw_train_ds =tf.keras.preprocessing.text_dataset_from_directory(os.getcwd()+'\\train', shuffle = False,seed=seed)
raw_test_ds =tf.keras.preprocessing.text_dataset_from_directory(os.getcwd()+'\\test', shuffle = False,seed=seed)

raw_val_ds = preprocessing.text_dataset_from_directory(
    os.getcwd()+'\\train',
    
    validation_split=0.2,
    subset='validation',seed=seed)

MAX_SEQUENCE_LENGTH = 250

int_vectorize_layer = TextVectorization(
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH)

train_text = raw_train_ds.map(lambda text, labels: text)

int_vectorize_layer.adapt(train_text)






def int_vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return int_vectorize_layer(text), label


text_batch, label_batch = next(iter(raw_train_ds))
first_question, first_label = text_batch[0], label_batch[0]



int_train_ds = raw_train_ds.map(int_vectorize_text)
int_val_ds = raw_val_ds.map(int_vectorize_text)
int_test_ds = raw_test_ds.map(int_vectorize_text)


AUTOTUNE = tf.data.AUTOTUNE
def configure_dataset(dataset):
  return dataset.cache().prefetch(buffer_size=AUTOTUNE)



int_train_ds = configure_dataset(int_train_ds)
int_val_ds = configure_dataset(int_val_ds)
int_test_ds = configure_dataset(int_test_ds)


def create_model(vocab_size, num_labels):
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, 128, mask_zero=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(128, 5,padding="valid", activation=tf.keras.activations.selu, strides=2),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(20, return_sequences=True),
    tf.keras.layers.LSTM(20),
    tf.keras.layers.Dropout(0.2),  
    tf.keras.layers.Dense(512),
    tf.keras.layers.Dropout(0.2),  
    tf.keras.layers.Dense(256),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


model = create_model(vocab_size=vocab_size + 1, num_labels=2)


model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer= tf.keras.optimizers.RMSprop(),metrics=['accuracy'])

history = model.fit(int_train_ds, validation_data=int_val_ds, epochs=6)

model.summary()


model.save_weights('weights')


model.save('model')



test_data=pd.read_csv('test_data.csv',index_col=None,usecols=['text','label'],dtype=dtype)

labels = test_data.pop('label')



test_data = tf.convert_to_tensor(test_data)
labels = tf.convert_to_tensor(labels)

##model = tf.keras.Sequential([int_vectorize_layer, model])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer= tf.keras.optimizers.RMSprop(),metrics=['accuracy'])


loss, accuracy = model.evaluate(test_data,labels)
print("Accuracy: {:2.2%}".format(accuracy))


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()




export_model = tf.keras.models.load_model('model')

export_model.load_weights('weights')


export_model = tf.keras.Sequential([int_vectorize_layer, export_model])



export_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer= tf.keras.optimizers.RMSprop(),metrics=['accuracy'])





export_model.call(test_data)




export_model.summary()


loss, accuracy = export_model.evaluate(test_data,labels)
print("Accuracy: {:2.2%}".format(accuracy))




def get_string_labels(predicted_scores_batch):
  predicted_int_labels = tf.argmax(predicted_scores_batch, axis=1)
  predicted_labels = tf.gather(raw_train_ds.class_names, predicted_int_labels)
  return predicted_labels


inputs = ["Says Tennessee is providing millions of dollars to virtual school company for results at the bottom of the bottom.	education,state-budget	andy-berke	Lawyer and state senator	Tennessee	democrat	0	0	0	0	0	a letter to state Senate education committee chairwoman Dolores Gresham."]

          

    
predicted_scores = export_model.predict(inputs)
predicted_labels = get_string_labels(predicted_scores)
for input, label in zip(inputs, predicted_labels):
  print("Question: ", input)
  print("Predicted label: ", label.numpy())
