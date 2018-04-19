# coding=utf-8
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import matplotlib.pyplot as plt

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()

# Set up logging
tf.logging.set_verbosity(tf.logging.DEBUG)
tf.logging.set_verbosity(tf.logging.INFO)
sess = tf.InteractiveSession()

# Load in the training data
# all_design.pkl
#
# R - total reflected light (normalized)
# T - transmission (normalized)
# A - absorption (normalized)
# d1-d8 - thickness of layer (Si, SiO2, alternating) (nm, I assume)
# lambda - wavelength of input light
data = pd.read_pickle('../all_design.pkl')
data = data.dropna()
data = data.sample(frac=1).reset_index(drop=True) # Shuffle the data for training purpose
data=(data-data.mean())/data.std() # Normalize the data 

# For all_design.pkl forward training output, select labels R, T, A
output_labels = ['R','T','A']

# Set up a diff function
diff = lambda l1,l2: [x for x in l1 if x not in l2]

# Grab the training labels
input_labels = diff(list(data.columns),output_labels)


# Now sort into train, validation, test sets
validation_size = int(np.floor(len(data)/10))
test_size = int(np.floor(len(data)/10))
train_size = 10000#len(data) - validation_size - test_size

validation = data.ix[0:(validation_size-1)]
test = data.ix[validation_size:(validation_size+test_size-1)]
train = data.ix[(validation_size+test_size):(validation_size+test_size+train_size)]

layersizes={
'dense1':500,
'dense2':400,
'dense3':200,
'dense4':100,
'dense5':50,
'dense6':25,
}

def dense_forward(inputs, layersizes, activation):
    iplayer='inputs'
    for k, val in layersizes.items():
        exec(k+' = tf.layers.dense('+iplayer+', val, activation=activation)')
        iplayer=k

    lasthidden=list(layersizes.keys())[-1]
    # Output layer
    output = tf.layers.dense(eval(lasthidden), len(output_labels), activation=activation)
    
    return output

n_epochs = 10
batch_size = 512
lr = 0.001

layer_inputs = tf.placeholder(tf.float32, (None, len(input_labels)))
layer_labels = tf.placeholder(tf.float32, (None, len(output_labels)))

layer_outputs = dense_forward(layer_inputs, layersizes, tf.nn.relu)
loss = tf.reduce_mean(tf.square(layer_outputs - layer_labels))  # claculate the mean square error loss
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

saver=tf.train.Saver()

init = tf.global_variables_initializer()

logs_path='zachmodel/'

with tf.Session() as sess:
    init.run()
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    
    global_step=0
    for epoch in range(n_epochs):
        print('Current Epoch:',epoch)
        test = test.sample(frac=1).reset_index(drop=True) # Shuffle the data for training purpose
        itertotal=train_size // batch_size
        for iteration in range(itertotal):
            
            feed_dict = {layer_inputs: test[input_labels].as_matrix()[(iteration*batch_size):(((iteration+1)*batch_size)-1)], layer_labels: test[output_labels].as_matrix()[(iteration*batch_size):(((iteration+1)*batch_size)-1)]}
            sess.run(train_op, feed_dict = feed_dict)

            hash=((60*iteration)//(itertotal-1))
            print("[{}{}] {}%".format('#' * hash, ' ' * (60-hash), int(iteration*100/(itertotal-1))), end="\r")
            
        global_step=global_step+itertotal
        print("\nTest Loss: {}".format(str(loss.eval(feed_dict={layer_inputs: test[input_labels].as_matrix(), layer_labels: test[output_labels].as_matrix()}))))
            #print("Layer 6 weights: {}".format(str(tf.get_variable("dense6/kernel"))))
        #print("Training Loss: {}".format(str(loss.eval(feed_dict={layer_inputs: train[input_labels].as_matrix(), layer_labels: train[output_labels].as_matrix()}))))
        #feed_dict = {layer_inputs: test[input_labels].as_matrix(), layer_labels: test[output_labels].as_matrix()}
        #sess.run(train_op, feed_dict=feed_dict)

        saver.save(sess,'./zachmodel_test/t1.ckpt',global_step=global_step)