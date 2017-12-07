from sklearn.neural_network import MLPClassifier
from read_file import read_files
from simple_classifier import generate_bow, create_bow_by_dict
train_set_raw = read_files('training.csv')
test_set_raw = read_files('test.csv')
train_set_features, word_dict = generate_bow(train_set_raw['x'], train_set_raw['label'])
test_set_features = create_bow_by_dict(test_set_raw['x'], word_dict)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(train_set_features, train_set_raw['label'])

prediction_list = clf.predict(test_set_features)
index = 0
hit = 0
miss = 0
print(prediction_list)
print (test_set_raw['label'])
for prediction in prediction_list:
    if test_set_raw['label'][index] == prediction:
        hit += 1
    else:
        miss += 1
print("hit is " + str(hit))
print("miss is " + str(miss))
print ("ratio is " + str(hit / (hit+miss)))