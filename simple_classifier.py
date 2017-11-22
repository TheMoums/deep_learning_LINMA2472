import xlrd
import string
from numpy import *
#from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords # Import the stop word list
from collections import Counter

# stemmer = nltk.LancasterStemmer()
stemmer = nltk.SnowballStemmer("english", ignore_stopwords=True)  # Better stemming
##############################################

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
             'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']


#############################
percentage = 0.05

list_trump_train = create_training_set("tweets_DonaldTrump_2009-2017_16k.xlsx",percentage)
list_trump_val = create_validation_set("tweets_DonaldTrump_2009-2017_16k.xlsx",percentage)


list_hillary_train = create_training_set("tweets_HillaryClinton_2013-2017_4k.xlsx",percentage)
list_hillary_val = create_validation_set("tweets_HillaryClinton_2013-2017_4k.xlsx",percentage)


words_trump = extract_words(list_trump_train, stopwords)
words_hillary = extract_words(list_hillary_train, stopwords)

nbr_of_words_trump = len(words_trump)
nbr_of_words_hillary = len(words_hillary)

most_common_words_trump = Counter(words_trump).most_common()
most_common_words_hillary = Counter(words_hillary).most_common()

occurency_dict_trump = normalize(most_common_words_trump, nbr_of_words_trump)
occurency_dict_hillary = normalize(most_common_words_hillary, nbr_of_words_hillary)

bag_size = 100  # Maximum bag of words size
significant_difference = 0.002  # the difference of usage becomes significant when reaching 0.001 % ? To discuss
filtered_trump = words_filter(most_common_words_trump, occurency_dict_trump, occurency_dict_hillary, bag_size, significant_difference)
filtered_hillary = words_filter(most_common_words_hillary, occurency_dict_hillary, occurency_dict_trump, bag_size, significant_difference)

print (filtered_trump)
print (most_common_words_trump)
#print (filtered_hillary)

"""
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 200)
bow = vectorizer.fit_transform(words)
print (vectorizer.get_feature_names())
# Numpy arrays are easy to work with, so convert the result to an
# array
bow = bow.toarray()
print (bow[0])"""
