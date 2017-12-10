#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 21:49:22 2017

Convolutionnal classifier using an embedding  for the words
    and Keras 
@author: brieuc
"""

"""
    Data Manipulation
"""
import numpy as np
import os
import csv
import gc
from gensim.models import KeyedVectors

if not 'embeddingModel' in vars():
    embeddingModel = 0
    gc.collect()
    embeddingModel = KeyedVectors.load_word2vec_format(os.environ['HOME']+'/Documents/Word2Vec_embedding/GoogleNews-vectors-negative300.bin', binary=True)#, norm_only=True)

def embedding(tweet):
    """ convert a tweet to a matrix
        with the embedding from Word2Vec to GoogleNews
    """
    E = []
    words = tweet.split()
    for word in words:
        if word in embeddingModel:
            E.append(embeddingModel[word])
    
    return np.array(E)
        

def create_dataset(filename):
    training_list = []
    label_list = []
    file = open(filename, "r")
    reader = csv.reader(file, delimiter=';')
    for tweet, author in reader:
        E = embedding(tweet)
        if not E.size<3*300:
            training_list.append(E)
            label_list.append(int(author=='Trump'))
    file.close()

    return {'x': training_list, 'label': label_list}

Train_dataset = create_dataset('training.csv')
x_train = Train_dataset['x']
y_train = Train_dataset['label']

Test_dataset = create_dataset('test.csv')
x_test = Test_dataset['x']
y_test = Test_dataset['label']

#what is the length of the maximal sequence of words (for padding)
seq_length = max(max([x.shape[0] for x in x_train]), max([x.shape[0] for x in x_test]))

def zero_padding(X):
    for i in range(len(X)):
        X[i] = np.vstack((X[i], np.zeros((seq_length-X[i].shape[0],300))))

zero_padding(x_train)
zero_padding(x_test)

x_train = np.array(x_train)
#print(x_train.shape)
x_test = np.array(x_test)
#print(x_test.shape)




"""
    Model definition and optimization
"""
from keras.models import Sequential,Model
from keras.layers import Input,Dense
from keras.layers import Conv1D, GlobalMaxPooling1D,Concatenate
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras import regularizers

"""first simple definition"""
#model definition
print("First model : ")
#architecture
model = Sequential()
model.add(Conv1D(128, 3, activation='relu', input_shape=(seq_length,300)))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))


#loss function and optimizer
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#optimization with early stopping
earlyStopping = EarlyStopping(monitor='val_acc', patience=0, verbose=0, mode='auto')
model.fit(x_train, y_train, batch_size=50, epochs=10, callbacks=[earlyStopping], 
          validation_split=0.1, shuffle=True)

score = model.evaluate(x_test,y_test, batch_size=64)

#display accuracy
print("\nAccuracy on the test set : "+str(score[1])+"\n\n")


"""second definition, closer to the article,
    with multiple kernel sizes, Dropout and a l2 regularizer"""
#model definition
print("Second model : ")
#definition of a convolutionnal layer
# with different kernel size
inp = Input(shape=(seq_length,300))
convs = []
#1
conv = Conv1D(100, 3, activation='relu')(inp)
pool = GlobalMaxPooling1D()(conv)
convs.append(pool)
#2
conv = Conv1D(100, 4, activation='relu')(inp)
pool = GlobalMaxPooling1D()(conv)
convs.append(pool)
#3
conv = Conv1D(100, 5, activation='relu')(inp)
pool = GlobalMaxPooling1D()(conv)
convs.append(pool)
out = Concatenate()(convs)

conv_model = Model(inputs=inp, outputs=out)

#architecture
model = Sequential()
#model.add(Conv1D(128, 3, activation='relu', input_shape=(seq_length,300)))
#model.add(GlobalMaxPooling1D())
model.add(conv_model)
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))


#loss function and optimizer
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#optimization with early stopping
earlyStopping = EarlyStopping(monitor='val_acc', patience=0, verbose=0, mode='auto')
model.fit(x_train, y_train, batch_size=50, epochs=10, callbacks=[earlyStopping], 
          validation_split=0.1, shuffle=True)

score = model.evaluate(x_test,y_test, batch_size=64)

#display accuracy
print("\nAccuracy on the test set : "+str(score[1])+"\n\n")


"""third definition, with one dense layer added at the end"""
#model definition
print("Third model : ")
#definition of a convolutionnal layer
# with different kernel size
inp = Input(shape=(seq_length,300))
convs = []
#1
conv = Conv1D(100, 3, activation='relu')(inp)
pool = GlobalMaxPooling1D()(conv)
convs.append(pool)
#2
conv = Conv1D(100, 4, activation='relu')(inp)
pool = GlobalMaxPooling1D()(conv)
convs.append(pool)
#3
conv = Conv1D(100, 5, activation='relu')(inp)
pool = GlobalMaxPooling1D()(conv)
convs.append(pool)
out = Concatenate()(convs)

conv_model = Model(inputs=inp, outputs=out)

#architecture
model = Sequential()
#model.add(Conv1D(128, 3, activation='relu', input_shape=(seq_length,300)))
#model.add(GlobalMaxPooling1D())
model.add(conv_model)
model.add(Dropout(0.5))
model.add(Dense(20, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))


#loss function and optimizer
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#optimization with early stopping
earlyStopping = EarlyStopping(monitor='val_acc', patience=0, verbose=0, mode='auto')
model.fit(x_train, y_train, batch_size=50, epochs=10, callbacks=[earlyStopping], 
          validation_split=0.1, shuffle=True)

score = model.evaluate(x_test,y_test, batch_size=64)

#display accuracy
print("\nAccuracy on the test set : "+str(score[1])+"\n\n")


"""fourth definition, with two stacked LSTM"""
#model definition
print("Fourth model : ")
#architecture
model = Sequential()
#model.add(Conv1D(128, 3, activation='relu', input_shape=(seq_length,300)))
#model.add(GlobalMaxPooling1D())
model.add(LSTM(64, return_sequences=True,input_shape=(seq_length,300)))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

#loss function and optimizer
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#optimization with early stopping
earlyStopping = EarlyStopping(monitor='val_acc', patience=0, verbose=0, mode='auto')
model.fit(x_train, y_train, batch_size=50, epochs=20, callbacks=[earlyStopping], 
          validation_split=0.1, shuffle=True)

score = model.evaluate(x_test,y_test, batch_size=64)

#display accuracy
print("\nAccuracy on the test set : "+str(score[1])+"\n\n")