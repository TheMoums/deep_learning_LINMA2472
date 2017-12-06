#  -*- coding: utf-8 -*-
import re
import string
from numpy import *
import numpy as np
import nltk
from collections import Counter, defaultdict
import unicodedata


stemmer = nltk.SnowballStemmer("english", ignore_stopwords=True)  # Better stemming


def remove_control_chars(s):
    return control_char_re.sub('', s)

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

def create_bag_of_word(training_set, dict_trump, dict_hillary):
    list_bag_of_word = []
    for tweet, author in training_set:
        word_list = tweet.split()
        if author == 'Trump':

            bag_of_word = dict_trump.keys()
        else:
            bag_of_word = dict_hillary.keys()
        for word in word_list:
            if word in bag_of_word:
                pass
        print (bag_of_word)
        list_bag_of_word.append(bag_of_word)
    print(list_bag_of_word)
    return list_bag_of_word


############################
# tweets = sys.argv[1]
file = open("training.csv", "r")
training_list = []
list_trump_train = []
list_hillary_train = []
for line in file:
    print(line)
    training_list.append(tweet)
    if author == 'Trump':
        list_trump_train.append((tweet, author))
    else:
        list_hillary_train.append((tweet, author))
file.close()

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

create_bag_of_word(training_list, filtered_trump, filtered_hillary)

"""print ("Filtered Trump : " + str(filtered_trump) + "\n")
print ("Filtered hillary : " + str(filtered_hillary) + "\n")"""


