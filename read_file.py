import csv


def read_files(filename):
    training_list = []
    label_list = []
    file = open(filename, "r")
    reader = csv.reader(file, delimiter=';')
    for tweet, author in reader:
        training_list.append(tweet)
        label_list.append(author)
    file.close()

    return {'x': training_list, 'label': label_list}
