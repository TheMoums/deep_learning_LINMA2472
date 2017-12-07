import tensorflow as tf
from   tensorflow.examples.tutorials.mnist import input_data
import shutil
import os

class DeepNeuralNetwork:
    # Reference : 
    # https://www.tensorflow.org/get_started/mnist/beginners
    # http://web.stanford.edu/class/cs20si/lectures/notes_05.pdf
 
    def __init__(self, train_set, test_set, nb_classes, name = 'DNN'):
        
        self.train_set = train_set
        self.test_set = test_set
        self.nb_classes = nb_classes
        
        self.is_running = False
        self.model_name = name

        
        # TF saver to save regularly our progress
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name = 'global_step')
        
        self.saver =  tf.train.Saver()

    def create_model(self, hidden_layers):
        # At least one hidden layer !
        # Create the structure of the deep neural network
       
        input_layer_size = len(self.train_set['x'][0]) 
        output_layer_size = self.nb_classes

        self.x = tf.placeholder(tf.float32, [None, input_layer_size])
        self.y_ = tf.placeholder(tf.float32, [None, output_layer_size])
        
        self.hidden_layers = hidden_layers 

        self.weights = []
        self.layers = []
        self.bias = []
        
        W = tf.Variable(tf.random_normal([input_layer_size, self.hidden_layers[0]], stddev=0.1))
        self.weights.append(W) 
        b = tf.Variable(tf.zeros([self.hidden_layers[0]]))
        self.bias.append(b)
        y = tf.nn.sigmoid(tf.matmul(self.x, W)+b)
        self.layers.append(y)
        
        for i in range(len(self.hidden_layers)-1):
            W = tf.Variable(tf.random_normal([self.hidden_layers[i], self.hidden_layers[i+1]], stddev=0.1))
            self.weights.append(W) 
            b = tf.Variable(tf.zeros([self.hidden_layers[i+1]]))
            self.bias.append(b)
            y = tf.nn.sigmoid(tf.matmul(self.layers[i], W)+b)
            self.layers.append(y)

        W = tf.Variable(tf.random_normal([self.hidden_layers[-1], output_layer_size], stddev=0.1))
        self.weights.append(W)
        b = tf.Variable(tf.zeros([output_layer_size]))
        self.bias.append(b)
        
        self.y = tf.matmul(self.layers[-1], W)+b

        # Loss function
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y_, logits = self.y))

        # Training accuracy
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.training_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        # Optimizer
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss, global_step = self.global_step)
    
    def train(self):
        
        if not self.is_running:
            self.run()
        losses = []
        total_loss = 0
        for step in range(10000):
            batch_xs, batch_ys = mnist.train.next_batch(30)
            loss_batch, _, summary = self.sess.run([self.loss, self.train_step, self.summary_op], feed_dict={self.x:batch_xs, self.y_:batch_ys})
            self.writer.add_summary(summary, global_step = step)
            total_loss += loss_batch    
            if step % 1000 == 0: 
                self.saver.save(self.sess, 'checkpoints/'+self.model_name, global_step=self.global_step) 
            losses.append(total_loss) 
            total_loss = 0

    def evaluate(self, x):
        if not self.is_running:
            self.run()

        return self.sess.run(tf.argmax(self.y, 1), feed_dict= {self.x: x})  


    def test(self):   

        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        return self.sess.run(accuracy, feed_dict={self.x: self.test_set['x'], self.y_ : self.test_set['label']}) 
        
    
        
    def run(self):
        self.sess = tf.InteractiveSession()
        if os.path.exists('.graphs'):  
            shutil.rmtree('.graphs')
        
        self.writer = tf.summary.FileWriter('.graphs', self.sess.graph)    
        self.create_summaries()

        self.is_running = True
        tf.global_variables_initializer().run()
    
    def close(self):
        self.writer.close()
        self.sess.close()
        self.is_running = False

    def create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            self.summary_op = tf.summary.merge_all()
    
    def restore(self):
        if not self.is_running:
            self.run()
        #saver = tf.train.import_meta_graph('checkpoints/2_layers_perceptron-9001.meta')
        #saver.restore(self.sess,tf.train.latest_checkpoint('.'))
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path) 
            tf.global_variables_initializer().run() 
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print(DNN.sess.run(DNN.bias[0], feed_dict= {DNN.x: [test_set['x'][0]]}))

# example : 

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_set = {'x' : mnist.train.images, 'label' : mnist.train.labels}
test_set  = {'x' : mnist.test.images,  'label' : mnist.test.labels}

DNN = DeepNeuralNetwork(train_set, test_set, 10, name='2_layers_perceptron')
DNN.create_model([16])
DNN.run()
print(DNN.sess.run(DNN.bias[0], feed_dict= {DNN.x: [test_set['x'][0]]}))
#DNN.train()
DNN.restore()
print('accuracy : ' + str(DNN.test()))
print(DNN.evaluate([test_set['x'][0]]))
DNN.close()
