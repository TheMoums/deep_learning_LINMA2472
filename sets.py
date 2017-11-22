#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 16:54:26 2017

@author: romain
"""

import xlrd
import string
from numpy import *
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords # Import the stop word list
from collections import Counter
import openpyxl as xl

#double valid_set = 0.05
def create_validation_set(filename, valid_set=0.05):
    book = xlrd.open_workbook(filename)
    sheet = book.sheet_by_index(0)
    cells = sheet.col(colx=1)
    i = 1/valid_set
    k = 0
    set_val = []
    for c in cells:
        if k % i == 0:
            translator = str.maketrans('', '', string.punctuation)
            sentence = c.value.lower().translate(translator)
            #print('%f : %s \n',i,sentence)
            set_val.append(sentence)
        k+=1
    return set_val
    
    
    
    
def create_training_set(filename,valid_set = 0.05):
    book = xlrd.open_workbook(filename)
    sheet = book.sheet_by_index(0)
    cells = sheet.col(colx=1)
    i = 1/valid_set
    k = 0
    set_train = []
    for c in cells:
        if k % i != 0:
            translator = str.maketrans('', '', string.punctuation)
            sentence = c.value.lower().translate(translator)
            #print('%f : %s \n',i,sentence)
            set_train.append(sentence)
        k+=1
    return set_train
    
pc = 0.05
set1 = create_validation_set('tweets_HillaryClinton_2013-2017_4k.xlsx',pc)
set2 = create_training_set('tweets_HillaryClinton_2013-2017_4k.xlsx',pc)
print(set1[1])
print(set2[1])
print(len(set1),len(set2))