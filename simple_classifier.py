#  -*- coding: utf-8 -*-
import csv
import re
import string
from numpy import *
import numpy as np
import nltk
from collections import Counter, defaultdict
import unicodedata
import read_file


stemmer = nltk.SnowballStemmer("english", ignore_stopwords=True)  # Better stemming

##############################################


def extract_words(list_tweets, stopwords):
    list_words = []
    for tweet in list_tweets:
        list_words += tweet.split()
    return [stemmer.stem(w) for w in list_words if not stemmer.stem(w) in stopwords]  # Not optimal to stem two times


def normalize(occurency_list, total):
    dictionnary = {}
    for key, value in occurency_list:
        dictionnary[key] = value / total
    return dictionnary


def words_filter(sorted_list, dict_1, dict_2, size, threshold):
    """sorted_list is the list of words sorted by occurency.
    Returns a dictionary with the "size" first elements of sorted_list that are 'significantly' more used in dict_1 than
    in dict_2"""
    filtered_dict = {}
    for key, value in sorted_list:
        if len(filtered_dict) > size:
            break  # Not beautiful but working
        elif key not in dict_2 or abs(dict_1[key] - dict_2[key]) > threshold:
            filtered_dict[key] = dict_1[key]
    return filtered_dict


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

def create_bag_of_word(training_set, final_dict, tf_idf, dict_idf = None):
    list_bag_of_word = []
    for tweet in training_set:
        word_list = tweet.split()
        bag_of_word = dict.fromkeys(final_dict.keys(), 0)
        tweet_length = 0
        for word in word_list:
            stemmed_word = stemmer.stem(word)
            tweet_length += 1
            if stemmed_word in stopwords:
                tweet_length -= 1
            elif stemmed_word in bag_of_word:
                bag_of_word[stemmed_word] += 1
        value_list = list(bag_of_word.values())
        if tf_idf and tweet_length != 0:
            value_list[:] = [x / tweet_length for x in value_list]
        list_bag_of_word.append(value_list)
    return list_bag_of_word


"""def nbr_containing(training_list, word):
    for tweet in training_list:
        if word 


def update_dict_idf(training_list, dict_idf):
    pass"""
############################


def generate_bag_of_words(training_list, label_list, tf_idf=False):
    list_trump_train = []
    list_hillary_train = []
    index = 0
    for tweet in training_list:
        if label_list[index] == 'Trump':
            list_trump_train.append(tweet)
        else:
            list_hillary_train.append(tweet)
        index += 1

    words_trump = extract_words(list_trump_train, stopwords)
    words_hillary = extract_words(list_hillary_train, stopwords)

    nbr_of_words_trump = len(words_trump)
    nbr_of_words_hillary = len(words_hillary)

    most_common_words_trump = Counter(words_trump).most_common()
    most_common_words_hillary = Counter(words_hillary).most_common()

    occurency_dict_trump = normalize(most_common_words_trump, nbr_of_words_trump)
    occurency_dict_hillary = normalize(most_common_words_hillary, nbr_of_words_hillary)

    bag_size = 100  # Maximum bag of words size
    significant_difference = 0.000  # the difference of usage becomes significant when reaching 0.001 % ? To discuss
    filtered_trump = words_filter(most_common_words_trump, occurency_dict_trump, occurency_dict_hillary, bag_size, significant_difference)
    filtered_hillary = words_filter(most_common_words_hillary, occurency_dict_hillary, occurency_dict_trump, bag_size, significant_difference)

    final_dict = {}
    for d in (filtered_trump, filtered_hillary):
        final_dict.update(d)
    """if tf_idf:
        dict_idf = dict.fromkeys(final_dict.keys(), 0)
        update_dict_idf(training_list, dict_idf)
        bag_of_words = create_bag_of_word(training_list, final_dict, tf_idf, dict_idf)
    else:"""
    bag_of_words = np.array(create_bag_of_word(training_list, final_dict, tf_idf))
    return bag_of_words, final_dict


training_list, label_list = read_file.read_files()
print(generate_bag_of_words(training_list, label_list))

