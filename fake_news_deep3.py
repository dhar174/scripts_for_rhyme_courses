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
import random

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
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

vocab_size = 10000

dtype = {
    "text": 'str',
    "label": 'uint8',
}


random.seed(87)

seed= random.randint(0,100)

raw_train_ds =tf.keras.preprocessing.text_dataset_from_directory(os.getcwd()+'\\train',batch_size= 16, shuffle = True,validation_split=0.3,seed=seed,subset='training',)
##raw_test_ds =tf.keras.preprocessing.text_dataset_from_directory(os.getcwd()+'\\test', shuffle = False,seed=seed)

raw_val_ds = preprocessing.text_dataset_from_directory(os.getcwd()+'\\train',shuffle = True,batch_size= 16,seed=seed,validation_split=0.3,subset='validation')

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
##int_test_ds = raw_test_ds.map(int_vectorize_text)


AUTOTUNE = tf.data.AUTOTUNE
def configure_dataset(dataset):
  return dataset.cache().prefetch(buffer_size=AUTOTUNE)



int_train_ds = configure_dataset(int_train_ds)
int_val_ds = configure_dataset(int_val_ds)

##int_test_ds = configure_dataset(int_test_ds)


def create_model(vocab_size, num_labels):
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, 128, mask_zero=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5,padding="valid", activation=tf.keras.activations.selu, strides=2),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(20, return_sequences=True),
    tf.keras.layers.LSTM(20),
    tf.keras.layers.Dropout(0.2),  
    tf.keras.layers.Dense(512, activation=tf.keras.activations.selu),
    tf.keras.layers.Dropout(0.2),  
    tf.keras.layers.Dense(256, activation=tf.keras.activations.selu),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


model = create_model(vocab_size=vocab_size + 1, num_labels=2)

model.trainable = True

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer= tf.keras.optimizers.RMSprop(learning_rate=0.0001),metrics=tf.keras.metrics.BinaryAccuracy())

history = model.fit(int_train_ds, validation_data=int_val_ds, epochs=50)

model.summary()



model.save_weights('weights')


model.save('model')


##
##test_data=pd.read_csv('test_data.csv',index_col=None,usecols=['label'],dtype=dtype)
##
##labels = test_data.pop('label')

random.seed(87)


raw_test_ds =tf.keras.preprocessing.text_dataset_from_directory(os.getcwd()+'\\test', shuffle = False,seed=random.randint(0,100))
##raw_test_val_ds = preprocessing.text_dataset_from_directory(os.getcwd()+'\\test',validation_split=0.2,subset='validation',seed=random.randint(0,100))


##model = tf.keras.Sequential([int_vectorize_layer, model])

##model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer= tf.keras.optimizers.SGD(learning_rate=0.001),metrics=['accuracy'])




int_test_ds = raw_test_ds.map(int_vectorize_text)

int_test_ds = configure_dataset(int_test_ds)


loss, accuracy = model.evaluate(int_test_ds,callbacks=tf.keras.callbacks.BaseLogger())
print("Eval Loss: ",loss)
print("Evaluation Accuracy: {:2.2%}".format(accuracy))
print("History Accuracy: {:2.2%}".format(float(history.history['binary_accuracy'][-1])))
print("History Validation Accuracy: {:2.2%}".format(float(history.history['val_binary_accuracy'][-1])))


plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
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



export_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer= tf.keras.optimizers.SGD(learning_rate=0.0001),metrics=tf.keras.metrics.BinaryAccuracy())






export_model.trainable=False




loss, accuracy = export_model.evaluate(raw_test_ds,callbacks=tf.keras.callbacks.History(),use_multiprocessing=True, workers=4,steps=1000)

print("Reloaded Accuracy: {:2.2%}".format(accuracy))

export_model.summary()



def get_string_labels(predicted_scores_batch):
  predicted_int_labels = tf.argmax(predicted_scores_batch, axis=1)
  predicted_labels = tf.gather(raw_train_ds.class_names, predicted_int_labels)
  return predicted_labels


inputs = ["If you’re reading this, I’ll assume you’ve had a Reese’s peanut butter cup at least once in your lifetime. I mean, who hasn’t? Not only are they available year round, they also come in fun shapes around the holidays. Do Reese’s trees and Reese’s eggs sound familiar? If you’re a fan of the popular peanut butter cups, your mouth is probably salivating right about now. Unfortunately, a deeper look into Reese’s ingredients might make you question that last minute purchase at the checkout line. As delicious as they are, Reese’s peanut butter cups can be detrimental to your health."+
          "A Little Background"+
          "Reese’s peanut butter cups were invented in 1928 by Mr. Reese. He was a farmer and a shipping foreman for Milton S. Hershey. After inventing the sweet treat, Mr. Reese decided to quit the dairy farming business and start his own candy company in his basement. And the rest is history."+


 
          "Reese’s peanut butter cups come in many different shapes, sizes and varieties. Although the chocolate to peanut butter ratio seems like perfection, the other ingredients in the popular candy are cause for concern."+



          "Ingredients In Reese’s Peanut Butter Cups"+
          'Ingredients include: Milk chocolate, (milk, chocolate, sugar, cocoa butter, chocolate, no fat milk, milk fat, lactose, soy lecithin, PGPR), peanuts, sugar dextrose, salt, TBGQ and citric acid.'+

          "The most questionable ingredients are:"+

          "1. Soy Lecithin"+
          "Research has shown that as much as 93% of soy is genetically modified. Soy lecithin has been found to have detrimental effects on fertility and reproduction. It can cause behavioral and cerebral abnormalities. It has also been linked to breast cancer."+

          "2. PGPR"+
          "PGPR is short for polyglycerol polyricinoleate. The manufacturer of this popular candy replaced cocoa butter with PGPR to lower the cost of production. PGPR comes from castor beans and it’s used to reduce the viscosity of chocolate. It has been connected to gastrointestinal problems and allergic reactions in children."+

          "3. TBHQ"+
          "TBHQ stands for tertiary butylhdroquinone. It’s derived from petroleum and can be extremely toxic. Side effects of ingesting TBHQ include nausea, vomiting, ringing in the ears, delirium and collapse. Research has shown that TBHQ can damage the lungs and umbilical cells in humans. It can also cause stomach cancer. Children who are exposed to this chemical may show anxiety, restlessness and intensified ADHD symptoms."]

          

    
predicted_scores = export_model.predict(inputs)
predicted_labels = get_string_labels(predicted_scores)
for input, label in zip(inputs, predicted_labels):
  print("Question: ", input)
  print("Predicted label: ", label.numpy())

inputs = ["Democrats push forward with $1.9 trillion COVID bill, clearing Senate hurdle"+
'BY GRACE SEGERS'+

'FEBRUARY 3, 2021 / 7:05 AM / CBS NEWS'+


"Washington — The Senate on Tuesday cleared a procedural hurdle on the road to passing President Biden's $1.9 trillion coronavirus relief proposal, a signal that congressional Democrats will continue to move forward with a vote to provide more economic assistance whether or not Republicans come to the negotiating table.'"+


"'We cannot, cannot afford to dither, delay or dilute. We need a big, bold package along the lines of what President Biden has proposed, the American Relief Plan. We hope that our Republican colleagues will join us in offering amendments,' Senate Majority Leader Chuck Schumer said in a speech ahead of the vote on Tuesday."+

"A motion to proceed to debate over the budget resolution that serves as the vehicle for the aid package passed by a 50 to 49 vote along party lines in the Senate on Tuesday afternoon."+


"Democrats are eschewing the traditional method of passing legislation in the Senate, which requires 60 votes to end debate on most legislation, in favor of an expedited process known as budget reconciliation that allows legislation to pass with a simple majority of 51 votes. Schumer and House Speaker Nancy Pelosi filed a joint budget resolution on Monday, kickstarting the reconciliation process."+

"CHUCK SCHUMER"+
'Senate Majority Leader Chuck Schumer speaks during a news conference in the Capitol in Washington on Tuesday, February 2, 2021.'+
'CAROLINE BREHMAN/CQ-ROLL CALL, INC VIA GETTY IMAGES'+
'There will be up to 50 hours of debate over the resolution, followed by a series of votes on amendments known as a "vote-a-rama." Schumer said in an earlier speech on the Senate floor that "there will be a bipartisan open amendment process on the budget resolution this week." Any senator may file an amendment during the "vote-a-rama," but it must be budgetary in nature.'+

"We welcome your ideas, your input, your revisions. We welcome cooperation. There is nothing about the process of a budget resolution or reconciliation, for that matter, that forecloses the possibility of bipartisanship,' Schumer said. Despite this overture by Schumer, Senate Minority Leader Mitch McConnell said Tuesday that Democrats had chosen a partisan path by proceeding with the budget resolution."+

'Once the House and Senate pass the budget resolution, committees get to work on drafting the reconciliation language. The Senate Budget Committee will examine the language of the bill to make sure it complies with the so-called "Byrd Rule," which limits what can be in reconciliation legislation and bars material considered "extraneous."'+


'Debate on the reconciliation bill in the Senate will be limited to 20 hours, followed by another "vote-a-rama" in which senators can offer amendments and raise a point of order challenging provisions which they consider to be extraneous.'+

'However, the passage of the bill will be stalled by former President Trumps impeachment trial, which begins next week. According to Senate rules, the chamber may not consider any other issues while it is conducting a trial, meaning that the Senate will likely not be able to vote on a reconciliation bill until later this month or early March.'+

'The vote to begin debate on the budget resolution comes after Mr. Biden met with nine Republicans on Monday to discuss their far smaller COVID relief proposal. The group of Republicans, which does not include any members of party leadership, have proposed a $600 billion bill that is far more limited in scope. The GOP proposal does not include money for state and local governments, which has been a sticking point in past negotiations on relief measures.'+

'Republican senators met with Mr. Biden at the White House for over two hours on Monday. There was no agreement, but Senator Susan Collins said after the meeting that conversations would continue at the staff level.'+

"We're very appreciative, as his first official meeting in the Oval Office, that the president chose to spend so much time with us in a frank and very useful discussion, Collins said."+

"Schumer told reporters later on Tuesday that he believed President Biden is 'totally on board with using reconciliation.'"+

"That is what President Biden wants us to do, and that is what we're doing, Schumer said."+

'The president with Democratic senators Tuesday during their caucus luncheon to discuss the measure. A Democratic senator who was on the call told CBS News that Mr. Biden said he "welcomed the conversation" with Senate Republicans last night, but that he "wont forget" the middle class.'+

'First published on February 2, 2021 / 1:00 PM'+

'© 2021 CBS Interactive Inc. All Rights Reserved.'+

"Grace Segers"]

predicted_scores = export_model.predict(inputs)
predicted_labels = get_string_labels(predicted_scores)
for input, label in zip(inputs, predicted_labels):
  print("Question: ", input)
  print("Predicted label: ", label.numpy())
