import csv


def read_files():
    training_list = []
    label_list = []
    file = open("training.csv", "r")
    reader = csv.reader(file, delimiter=';')
    for tweet, author in reader:
        training_list.append(tweet)
        label_list.append(author)
    file.close()
    return training_list, label_list
