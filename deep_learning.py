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

        self.x = tf.placeholder(tf.float32, [None, input_layer_size], name='input')
        self.y_ = tf.placeholder(tf.float32, [None, output_layer_size], name = 'label')
        
        self.hidden_layers = hidden_layers 

        self.weights = []
        self.layers = []
        self.bias = []
        
        W = tf.Variable(tf.random_normal([input_layer_size, self.hidden_layers[0]], stddev=0.01))
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
        n_epoch = 10
 
        # Generate batches
        batches = batch_iter(
            list(zip(self.train_set['x'], self.train_set['label'])), 32, n_epoch)
        
        # Training loop. For each batch...
        step = 0
        total_loss = 0
        losses = []

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

