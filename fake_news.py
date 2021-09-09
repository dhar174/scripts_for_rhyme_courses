import itertools
import pandas as pd
import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

train_data=pd.read_csv('train_datax.csv', index_col='id')
test_data=pd.read_csv('test_data.csv',index_col='id')


        

x_train = train_data['text']

y_train = train_data['label'].to_frame()
x_test = test_data.drop('label', axis=1)
x_test = test_data['text']


y_test = test_data['label'].to_frame()

tfidf_vectorizer=TfidfVectorizer(stop_words='english',strip_accents='ascii')
x_train_list=[]
for row in x_train.to_numpy():
    x_train_list.append(str(row))

x_test_list=[]
for row in x_test.to_numpy():
    x_test_list.append(str(row))


##x_train=x_train.to_numpy()
##x_train = x_train.tolist()
##
##
##x_test=x_test.to_numpy()
##x_test = x_test.tolist()

##map(str, x_train)
##map(str, x_test)
##y_train = y_train.to_numpy()
##y_train = y_train.ravel()
x_test = x_test.to_numpy()
x_test = x_test.ravel()

print(x_train)
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

pac=PassiveAggressiveClassifier(max_iter=50)
print(y_train.shape)
print(tfidf_test.shape)

pac.fit(tfidf_train,y_train['label'].values)

print(y_test.count())

y_pred=pac.predict(tfidf_test)

print(type(y_pred))
print(y_pred.tolist())
score=accuracy_score(y_test.to_numpy(),y_pred.ravel())
print(f'Accuracy: {round(score*100,2)}%')

print(confusion_matrix(y_test,y_pred, labels=[0,1]))

# Vectorizing and applying TF-IDF
from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(tfidf_train,y_train['label'].values)



y_pred=LR.predict(tfidf_test)
# Accuracy
print("accuracy: {}%".format(round(accuracy_score(y_test, y_pred.ravel())*100,2)))
print(confusion_matrix(y_test,y_pred.ravel(), labels=[0,1]))

from sklearn.tree import DecisionTreeClassifier
DTC=DecisionTreeClassifier()
DTC.fit(tfidf_train,y_train['label'].values)



y_pred=DTC.predict(tfidf_test)
# Accuracy
print("accuracy: {}%".format(round(accuracy_score(y_test, y_pred.ravel())*100,2)))
print(confusion_matrix(y_test,y_pred.ravel(), labels=[0,1]))

from sklearn.ensemble import RandomForestClassifier
RFC=RandomForestClassifier()
RFC.fit(tfidf_train,y_train['label'].values)



y_pred=RFC.predict(tfidf_test)
# Accuracy
print("accuracy: {}%".format(round(accuracy_score(y_test, y_pred.ravel())*100,2)))
print(confusion_matrix(y_test,y_pred.ravel(), labels=[0,1]))

