import csv
import re
import xlrd
import random
from collections import Counter, defaultdict
from numpy import *


def create_validation_set(filename, valid_set=0.05):
    book = xlrd.open_workbook(filename)
    sheet = book.sheet_by_index(0)
    cells = sheet.col(colx=1)
    i = 1 / valid_set
    k = 0
    set_val = []
    for c in cells:
        if k % i == 0:
            tweet = c.value.lower()
            tweet = re.sub(r"http\S+", "", tweet)  # Remove URL
            clean = re.sub('[^a-zA-Z0-9]+', ' ', tweet)  # Keep just letters and numbers
            set_val.append(clean)
        k += 1
    return set_val


def create_training_set(filename, valid_set=0.05):
    book = xlrd.open_workbook(filename)
    sheet = book.sheet_by_index(0)
    cells = sheet.col(colx=1)
    i = 1 / valid_set
    k = 0
    set_train = []
    for c in cells:
        if k % i != 0:
            tweet = c.value.lower()
            tweet = re.sub(r"http\S+", "", tweet)  # Remove URL
            clean = re.sub('[^a-zA-Z0-9]+', ' ', tweet)  # Keep just the letters
            set_train.append(clean)
        k += 1
    return set_train


def make_tuple(list1,list2):
    if len(list1) == len(list2):
        newlist = []
        for i in range(0,len(list1)):
            newlist.append((list1[i], list2[i]))
        return newlist
    else:
        return None

def create_training_test_set():    
    percentage = 0.05

    list_trump_train = create_training_set("tweets_DonaldTrump_2009-2017_16k.xlsx", percentage)
    list_hillary_train = create_training_set("tweets_HillaryClinton_2013-2017_4k.xlsx", percentage)
    label_train_trump = ['Trump'] * len(list_trump_train)
    label_train_hillary = ['Clinton'] * len(list_hillary_train)

    list_trump_val = create_validation_set("tweets_DonaldTrump_2009-2017_16k.xlsx", percentage)
    list_hillary_val = create_validation_set("tweets_HillaryClinton_2013-2017_4k.xlsx", percentage)
    label_val_trump = ['Trump'] * len(list_trump_val)
    label_val_hillary = ['Clinton'] * len(list_hillary_val)
    training_set = list_trump_train + list_hillary_train
    label_train_all = label_train_trump + label_train_hillary
    validation_set = list_trump_val + list_hillary_val
    label_val_all = label_val_trump + label_val_hillary

    training_set = make_tuple(training_set, label_train_all)
    validation_set = make_tuple(validation_set, label_val_all)
    random.shuffle(training_set)
    random.shuffle(validation_set)
    
    file = open("training.csv", "w")
    writer = csv.writer(file, delimiter=';', lineterminator='\n')
    for (tweet, author) in training_set:
        writer.writerow([tweet, author])
    file.close()
    file = open("test.csv", "w")
    writer = csv.writer(file, delimiter=';', lineterminator='\n')
    for (tweet, author) in validation_set:
        writer.writerow([tweet, author])
    file.close()
    
