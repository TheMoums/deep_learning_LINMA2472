import tensorflow as tf
import numpy as np

# https://www.tensorflow.org/get_started/estimator



### HYPERPARAMETERS 
def create_bow_model(features, label, nb_words=100):
    bow = tf.contrib.layers.bow_encoder(features, vocab_size=nb_words, embed_dim=1)
    return bow

#with np.load("/var/data/training_data.npy") as data:
with np.load("tweets_DonaldTrump_2009-2017_16k.xlsx") as data:
    features = data["features"]
    labels = data["label"]

assert features.shape[0] == labels.shape[0]

dataset = tf.data.Dataset.from_tensor_slices((features, label))

### CREATE MODEL

feature_columns =[tf.feature_column.numeric_column("x", shape=[1])]

classifier = tf.estimator.DNNClassifier(
    feature_columns = feature_columns
    hiddent_units = [100]
    optimizer=tf.train.ProximalAdagradOptimizer(
    learning_rate=0.1,
    l1_regularization_strength=0.001)
    )

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(training_set.data)},
    y=np.array(training_set.target),
    num_epochs=None,
    shuffle=True)

### OPTIMIZE
classifier.train(input_fn = train_input_fn, steps=2000)

### TEST
# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(test_set.data)},
    y=np.array(test_set.target),
    num_epochs=1,
    shuffle=False)
    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
    )
#x = tf.placeholder(tf.float32, [None, nb_words])
#y_ = tf.placeholder(tf.float32, [None, 2])

#W = tf.Variable(tf.random_normal([784, 50], stddev=0.1))
#W1 = tf.Variable(tf.rand
