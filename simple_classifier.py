#  -*- coding: utf-8 -*-
import csv
import re
import string
from numpy import *
import numpy as np
import nltk
from collections import Counter, defaultdict
import read_file


stemmer = nltk.SnowballStemmer("english", ignore_stopwords=True)  # Initialize stemmer

##############################################


def extract_words(list_tweets, stopwords):
    """Extract the words for the tweets. These are stemmed and ignored if stopword"""
    list_words = []
    for tweet in list_tweets:
        list_words += tweet.split()
    return [stemmer.stem(w) for w in list_words if not stemmer.stem(w) in stopwords]  # Not optimal to stem two times


def normalize(occurency_list, total, bag_size):
    """Normalize the word occurency by the number of tweets from its author"""
    dictionnary = {}
    for key, value in occurency_list:
        if len(dictionnary) > bag_size:
            break
        else:
            dictionnary[key] = value / total
    return dictionnary


stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
             'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
             'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
             'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
             'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
             'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
             'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
             'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
             'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
             'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
             'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn',
             'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', 'make', 'go']


#############################

def create_bow_by_dict(training_set, final_dict, tf_idf=False):
    """Internal function to to create the bow.
     Apply tfidf if tf_idf is set to True"""
    list_bag_of_word = []
    for tweet in training_set:
        word_list = tweet.split()
        bag_of_word = dict.fromkeys(final_dict.keys(), 0)
        tweet_length = 0  # the length of the tweet
        for word in word_list:
            stemmed_word = stemmer.stem(word)
            tweet_length += 1
            if stemmed_word in stopwords:
                tweet_length -= 1  # The length is not incremented if we encountered a stopword
            elif stemmed_word in bag_of_word:
                bag_of_word[stemmed_word] += 1  # Simple incremental bag of word
        value_list = list(bag_of_word.values())
        if tf_idf and tweet_length != 0:
            key_list = list(bag_of_word.keys())
            for i in range(0, len(value_list)):
                value_list[i] = (value_list[i] / tweet_length)*final_dict[key_list[i]]  # Apply tfidf formula on all words
        list_bag_of_word.append(value_list)  # Add the list to create the bag of word
    return list_bag_of_word


def create_dict_idf(list_trump, list_hillary, final_dict, length):
    """Creates a dict that assigns an IDF value to each word"""
    keys = final_dict.keys()
    dict_idf = dict.fromkeys(keys, 0)  # The dict we will return
    keys = list(keys)
    for key, value in list_trump:  # Assign to dict[key] the number of tweets where the word 'key' appears
        if key in keys:
            dict_idf[key] += value
    for key, value in list_hillary:
        if key in keys:
            dict_idf[key] += value
    for elem in list(dict_idf.keys()):
        dict_idf[elem] = np.log10(length/(dict_idf[elem]))  # Apply idf definition
    return dict_idf
############################


def generate_bow(training_list, label_list, tf_idf=False):
    """Creates the bag of word"""
    list_trump_train = []
    list_hillary_train = []
    index = 0
    for tweet in training_list:
        if label_list[index] == 'Trump':
            list_trump_train.append(tweet)  # List of Trump's tweets
        else:
            list_hillary_train.append(tweet)  # List of Hillary's tweets
        index += 1

    words_trump = extract_words(list_trump_train, stopwords)  # Extract the words from the tweets
    words_hillary = extract_words(list_hillary_train, stopwords)

    nbr_of_words_trump = len(words_trump)  # Number of words Trump used
    nbr_of_words_hillary = len(words_hillary)

    most_common_words_trump = Counter(words_trump).most_common()  # List of Trump's tweets sorted by occurency
    most_common_words_hillary = Counter(words_hillary).most_common()  # List of Hillary's tweets sorted by occurency

    bag_size = 100  # Maximum bag of words size
    dict_trump = normalize(most_common_words_trump, nbr_of_words_trump, bag_size)  # Normalize the occurency
    dict_hillary = normalize(most_common_words_hillary, nbr_of_words_hillary, bag_size)

    final_dict = {}  # The most common words used by Hillary and Trump
    for d in (dict_trump, dict_hillary):
        final_dict.update(d)
    if tf_idf:
        final_dict = create_dict_idf(most_common_words_trump[0:bag_size+1], most_common_words_hillary[0:bag_size+1], final_dict, len(training_list))
        bag_of_words = create_bow_by_dict(training_list, final_dict, tf_idf)
    else:
        bag_of_words = np.array(create_bow_by_dict(training_list, final_dict, tf_idf))
    return bag_of_words, final_dict


"""training_raw = read_file.read_files("training.csv")
print(generate_bow(training_raw["x"], training_raw["label"], True))"""

