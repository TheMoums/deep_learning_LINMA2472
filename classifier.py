import tensorflow as tf
import os
import shutil
import numpy as np

class Classifier:
    # Reference : 
    # https://www.tensorflow.org/get_started/mnist/beginners
    # http://web.stanford.edu/class/cs20si/
 
    def __init__(self, train_set, test_set, nb_classes, name = 'DNN'):
        
        self.train_set = train_set
        self.test_set = test_set
        self.nb_classes = nb_classes
        
        self.is_running = False
        self.model_name = name
        
        # TF saver to save regularly our progress
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name = 'global_step')
        
        self.saver =  tf.train.Saver()

    
    def train(self, n_epoch=10):
        
        if not self.is_running:
            self.run()
 
        # Generate batches
        batches = batch_iter(
            list(zip(self.train_set['x'], self.train_set['label'])), 32, n_epoch)
        
        # Training loop. For each batch...
        step = 0
        total_loss = 0
        losses = []
        
        print('Training step :')
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            loss_batch, _, summary = self.sess.run([self.loss, self.train_step, self.summary_op], feed_dict={self.x : list(x_batch), self.y_ : list(y_batch)})
            self.writer.add_summary(summary, global_step = step)
            total_loss += loss_batch
            #print(loss_batch)
            if step % 100 == 0 :
                losses.append(total_loss)
                print('Step ' + str(step) + ' (' + str(total_loss /100) + ' mean loss)', end='\r')
                total_loss = 0
                self.saver.save(self.sess, 'checkpoints/'+self.model_name, global_step=self.global_step) 

            step += 1
        

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

def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    source : https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
    """
    
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
