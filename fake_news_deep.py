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


dtype = {
    "text": 'str',
    "label": 'uint8',
}
##train_data=pd.read_csv('train_data.csv', index_col=None,dtype=dtype)
##test_data=pd.read_csv('test_data.csv',index_col=None,dtype=dtype)

##with open('test_labels.csv') as f:
##    test_labels = f.read().splitlines()
##
##with open('train_labels.csv') as f:
##    train_labels = f.read().splitlines()

##test_data.dropna(inplace=True)
##train_data.dropna(inplace=True)

##print(len(test_data))

seed=42

raw_train_ds =tf.keras.preprocessing.text_dataset_from_directory(os.getcwd()+'\\train', shuffle = False,seed=seed)
raw_test_ds =tf.keras.preprocessing.text_dataset_from_directory(os.getcwd()+'\\test', shuffle = False,seed=seed)

raw_val_ds = preprocessing.text_dataset_from_directory(
    os.getcwd()+'\\train',
    
    validation_split=0.2,
    subset='validation',seed=seed)

##train_data=pd.read_csv('train_data.csv', index_col=None,usecols=['text','label'],dtype=dtype)
##test_data=pd.read_csv('test_data.csv',index_col=None,usecols=['text','label'],dtype=dtype)


##print(train_data)
##print(train_data.count())
####print(test_data[23])
##print(test_data.count())
##print(test_data)
##
##print(train_data.loc[8920])
##print(train_data.loc[166])
##
##print(train_data)
##train_data = train_data[~train_data.text.duplicated()]
##train_data.reset_index(drop=True,inplace=True)
##print(train_data.loc[8920])
##print(train_data.loc[166])
##test_data = test_data[~test_data.text.duplicated()]
##test_data.reset_index(drop=True,inplace=True)
##
##train_data.dropna(inplace=True)
##test_data.dropna(inplace=True)
##print(train_data)
##print(train_data.count())
##print(test_data.count())
##print(test_data)
##tokenizer = Tokenizer()
##tokenizer.fit_on_texts(train_data['text'].to_string(index=False))
##word_index = tokenizer.word_index
vocab_size = 10000

##print(vocab_size)
##
##
##print('columns_train: ',train_data.columns)
##print('columns_test: ',test_data.columns)

# Padding data

##sequences = tokenizer.texts_to_sequences(train_data['text'].to_string(index=False))
##padded = pad_sequences(sequences, maxlen=500, padding='post', truncating='post')
##
##
##print(type(sequences))
##print(sequences[0])
##print(sequences[0][0])


##x_train = []
##for i in train_data['text']:
##    if i:
##        x_train.append(i)
##
##x_train = np.asarray(x_train,dtype=object)
##
##y_train = train_data['label'].to_numpy()
##
##
####sequences_test = tokenizer.texts_to_sequences(test_data['text'].to_string(index=False))
####padded_test = pad_sequences(sequences_test, maxlen=500, padding='post', truncating='post')
##
##x_test = []
##for i in test_data['text']:
##    if i:
##        x_test.append(i)
##x_test = np.asarray(x_test,dtype=object)
##
##y_test = test_data['label'].to_numpy()
####print(x_train.values.astype(str))
##
####tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7, min_df=0.1)
####x_train_list=[]
####for row in x_train.to_numpy():
####    x_train_list.append(str(row))
####
####x_test_list=[]
####for row in x_test.to_numpy():
####    x_test_list.append(str(row))
##
####
####y_train = y_train.to_numpy()
####y_train = y_train.ravel()
####x_test = x_test.to_numpy()
####x_test = x_test.ravel()
##
embeddings_index = {};



MAX_SEQUENCE_LENGTH = 250

int_vectorize_layer = TextVectorization(
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH)


##print(x_train.shape)
##print(y_train.shape)
##print(x_test.shape)
##print(y_test.shape)
##
##print(x_train[0])
##print(y_train[0])
##print(x_test[0])
##print(y_test)
##t=-1
##
##
##
##for x in x_train:
##    t+=1
##    if type(x)!=str:
##        print("Baackaaaaw!")
##        print('Culprit: ',x,t)
##        break


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
##  model = tf.keras.Sequential([
##      layers.Embedding(vocab_size, 64, mask_zero=True),
##      layers.Conv1D(64, 5, padding="valid", activation="relu", strides=2),
##      layers.GlobalMaxPooling1D(),
##      layers.Dense(num_labels)
##  ])
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

# vocab_size is VOCAB_SIZE + 1 since 0 is used additionally for padding.
int_model = create_model(vocab_size=vocab_size + 1, num_labels=4)
##int_model.compile(
##    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
##    optimizer='adam',
##    metrics=['accuracy'])

int_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer= tf.keras.optimizers.RMSprop(),metrics=['accuracy'])

history = int_model.fit(int_train_ds, validation_data=int_val_ds, epochs=10)



int_model.summary()





##history = model.fit(train_data, epochs=5, batch_size=100, validation_data=[test_data])
exit()

vec_train = pd.DataFrame(dtype=object)
vec_test = pd.DataFrame(dtype=object)
idx=0
print(len(y_train))
print(len(x_train))
print(vec_train.shape)
##print(train_data.index.duplicated())
for i in x_train:
##    print(i)
##    print(type(i))
##    print(np.where(x_train==i)[0])
####    print(y_train[int(np.where(x_train==i)[0])])
##    print(idx)
    if i and int(np.where(x_train==i)[0]) == idx and i != ' ':
        a=i
        i = preprocessing.text.text_to_word_sequence(i)
##        print(i)
    else:
        print("QUARANTINEWHILE...")
##        train_data.drop(index=idx)
##        train_data.loc[idx].id = train_data.loc[idx].name
##        train_data.reset_index(inplace=True,drop=True)
##        np.delete(x_train, idx)
        i = None
        
    if i:
##        print(type(int_vectorize_text(i)))
##        print(int_vectorize_text(i))
        temp_train.insert(idx,int_vectorize_text(i))
        
        
    else:
        if(a != ' '):
            print(a,i)
            print("QUARANTINEWHILE...")
            np.delete(x_train, idx)
            train_data.drop(index=idx,inplace=True)
            train_data.reset_index(inplace=True,drop=True)
        else:
            np.delete(x_train, idx)
            train_data.drop(index=idx,inplace=True)
            train_data.reset_index(inplace=True,drop=True)
    idx+=1
        
idx=0
print(temp_train)
##train_data = tf.data.Dataset.from_tensor_slices(train_data.values)

vec_train = tf.convert_to_tensor(temp_train)
train_data['text'] = vec_train

temp_train, x_train = None

print(len(y_train))
print(len(x_train))
print(len(vec_train))
temp_test=[]
    
for i in x_test:
##    print(i)
##    print(idx)
##    print(x_test[idx])
    if i and int(np.where(x_test==i)[0]) == idx and i != ' ':
        a=i
        i = preprocessing.text.text_to_word_sequence(i)
##        print(i)
    else:
        print("QUARANTINEWHILE...",idx,int(np.where(x_test==i)[0]))
##        test_data.drop(index=idx)
##        test_data.loc[idx].id = test_data.loc[idx].name
##        test_data.reset_index(inplace=True,drop=True)
##        np.delete(x_train, idx)
        
        
    if i:
##        print(type(int_vectorize_text(i)))
##        print(int_vectorize_text(i))
        temp_test.insert(idx,int_vectorize_text(i))
        
        
    else:
        if(a != ' '):
##            print(a,i,idx,test_data.loc[idx],test_data.loc[idx+1])
            print("QUARANTINEWHILE...",idx)
    ##        vec_test.drop(vec_test.loc[int(np.where(x_test==a)[0])])
            np.delete(x_test, idx)
            print(type(test_data))
            print(test_data)
            
            test_data.drop(index=idx,inplace=True)
            test_data.reset_index(inplace=True,drop=True)
            
        else:
            np.delete(x_test, idx)
            test_data.drop(index=idx,inplace=True)
            test_data.reset_index(inplace=True,drop=True)
    idx+=1



##test_data = tf.data.Dataset.from_tensor_slices(test_data.values)

vec_test = tf.convert_to_tensor(temp_test)
test_data['text'] = vec_test
##try:
##    int_train_ds = vec_train.map(int_vectorize_text)
##    int_val_ds = raw_val_ds.map(int_vectorize_text)
##    int_test_ds = vec_test.map(int_vectorize_text)
##except:
##    print(int_vectorize_text)


temp_test, x_test = None

embeddings_matrix = np.zeros((vocab_size+1, 100));
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word);
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector;


##tfidf_train=tfidf_vectorizer.fit_transform(x_train_list) 
##tfidf_test=tfidf_vectorizer.transform(x_test_list)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, 100, weights=[embeddings_matrix], trainable=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(20, return_sequences=True),
    tf.keras.layers.LSTM(20),
    tf.keras.layers.Dropout(0.2),  
    tf.keras.layers.Dense(512),
    tf.keras.layers.Dropout(0.3),  
    tf.keras.layers.Dense(256),
    tf.keras.layers.Dense(1, activation='sigmoid')
])



print("Training Complete")

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
