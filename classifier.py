import tensorflow as tf
import os
import shutil
import numpy as np

class Classifier:
    # Implementation of an abstraction of a classifier : 
    #  - Initialize the classifier
    #  - Training step
    #  - Evaluation method to determine the estimated class of an input (after the training)
    #  - Test step
    #
    # Utilitaries to help the use of the model :
    #  - Create summaries to track the progress of the training through tensorboard
    #  - Save the model parameters regularly
    #
    # References :
    #
    # Those two resources greatly helped us to build this code :
    #
    # https://www.tensorflow.org
    # http://web.stanford.edu/class/cs20si/
    #

    def __init__(self, train_set, test_set, nb_classes, name = 'DNN'):
        # Initialize the classifier
        # Input :
        #   - train_set  : a dictionary whose key 'x' gives the features of each input and the key 'label' gives the label of each input
        #   - test_set   : a dictionary having the same structure
        #   - nb_classes : number of class the classifier has to discriminate
        #   - name       : name of the model
        #
        
        self.train_set = train_set
        self.test_set = test_set
        self.nb_classes = nb_classes
        
        self.is_running = False
        self.model_name = name
        
        # TF saver to save regularly our progress
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name = 'global_step')
        self.saver =  tf.train.Saver()
   
    
    def create_model():
        # Model of the classifier.  Please extend this class to create a classifier 
        # You should provide :
        #   - a model mapping self.x (the input) to self.y (the estimated class)
        #   - a tensorflow loss function (self.loss) of the model as a function of self.y_ and self.Y
        #   - a tensofrlow optimizer (self.train_step) using self.loss 
        #
        # Example :
        # 
        # weights = tf.Variable(tf.zeros([len(self.x), len(self.nb_classes)])) 
        # self.y = tf.multiply(self.x, weights)
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y_, logits = self.y))
        # self.train_step = tf.train.AdamOptimizer().minimize(self.loss, global_step = self.global_step)
        #

        pass
    
    def train(self, n_epoch=10, batch_size = 32):
        # Train the model, passing n_epoch times through the training_set
        # The train set is divided into batch of size batch_size and at each
        # step, the model is optimized with respect to this batch using the
        # the optimizer provied in the model.
        # Run a TF session if not already done
        if not self.is_running:
            self.run()
 
        # Generate batches
        batches = batch_iter(
            list(zip(self.train_set['x'], self.train_set['label'])), batch_size, n_epoch)
        
        # Training loop : for each batch, optimize the model with respect to this batch
        step = 0
        total_loss = 0
        losses = []
        
        print('Training step :')
        for batch in batches:

            x_batch, y_batch = zip(*batch)
            
            # Run the train step 
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

        return losses 

    def evaluate(self, x):
        # Evaluate a list of input (even if you want to evaluate only one input, put it in a list) using
        # the model.  The model should be create before, and either trained, either restored from a saved
        # version (using self.restore())

        if not self.is_running:
            self.run()

        return self.sess.run(tf.argmax(self.y, 1), feed_dict= {self.x: x})  


    def test(self):   
        # Evaluate the model using the data in self.test_set
        
        if not self.is_running:
            self.run()

        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        return self.sess.run(accuracy, feed_dict={self.x: self.test_set['x'], self.y_ : self.test_set['label']}) 
        
    def run(self):
        # Run a TF session to train and use the model

        self.sess = tf.InteractiveSession()

        # Erase the save directory if it exists already 
        if os.path.exists('.graphs'):  
            shutil.rmtree('.graphs')
        
        # Create summaries to track the progress of the training on tensorboard
        self.writer = tf.summary.FileWriter('.graphs', self.sess.graph)    
        self.create_summaries()

        self.is_running = True
        tf.global_variables_initializer().run()
    
    def close(self):
        # Close the TF interactive Session 

        self.writer.close()
        self.sess.close()
        self.is_running = False

    def create_summaries(self):
        # Create summaries to track the progress of the training on tensorboard
        # At the moment, only the loss function can be tracked, but it can be 
        # extended to save the accuracy on the validation set for example
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def restore(self):
        # Restore an old save of the model 
        
        if not self.is_running:
            self.run()
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            tf.global_variables_initializer().run() 
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)


def batch_iter(data, batch_size, num_epochs):
    # Generates a batch iterator for a dataset that loop over the data num_epochs times and divide it
    # in a serie of batch of size batch_size.  The data is shuffled after each epoch.  If the size of 
    # the data is not a mutliple of batch_size, a final smaller batch is created with the remaining data.
    # reference : https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
    
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        
        for batch_num in range(num_batches_per_epoch):
            # Create a batch  
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
