# Imports
import tensorflow as tf
import numpy as np
import math as mat


# necessary constants
dim_input = 4683    # the dimensionality of our dataset
dim_hidden = 500    # number of second level nodes recommended by Van Der Maaten
dim_output = 4683   # nodes for reconstruction vector, obviously must match input vec

dim_input_ac_2 = 500
dim_hidden_ac_2 = 500
dim_output_ac_2 = 500

dim_input_ac_3 = 500
dim_hidden_ac_3 = 2000
dim_output_ac_3 = 500

dim_input_ac_4 = 2000
dim_hidden_ac_4 = 3
dim_output_ac_4 = 2000

learning_rate = 0.01
epoch = 25
batch_size = 100

# get our data as a numpy memmap
data = np.memmap('eboss.random.10000.memmap' , dtype='float32' , mode='r' , shape=(10000, 4683))

# dataset placeholder
X = tf.placeholder(tf.float32 , shape=(None, dim_input))

# we need to calculate a range for our uniform sampling
# the recommended value is U[-b , b] where b =  sqrt(6) / sqrt(dim_layer(i) + dim_layer(i-1))
dmax_ac_1 = mat.sqrt(6) / mat.sqrt(dim_input + 0)
dmin_ac_1 = dmax_ac_1 * -1.0

dmax_ac_2 = mat.sqrt(6) / mat.sqrt(dim_input_ac_2 + 0)
dmin_ac_2 = dmax_ac_2 * -1.0

dmax_ac_3 = mat.sqrt(6) / mat.sqrt(dim_input_ac_3 + 0)
dmin_ac_3 = dmax_ac_2 * -1.0

dmax_ac_4 = mat.sqrt(6) / mat.sqrt(dim_input_ac_4 + 0)
dmin_ac_4 = dmax_ac_2 * -1.0

# our weight matrix and vectors
W1 = tf.Variable(tf.random_uniform(shape=(dim_input , dim_hidden) , minval = dmin_ac_1 , maxval = dmax_ac_1, dtype=tf.float32))
#W2 = tf.transpose(W1) # this is done to more closely mimic structure of a RBM
W2 = tf.placeholder(tf.float32 , shape = (dim_hidden, dim_input))

W1_ac_2 = tf.Variable(tf.random_uniform(shape=(dim_input_ac_2 , dim_hidden_ac_2) , minval = dmin_ac_2 , maxval = dmax_ac_2, dtype=tf.float32))
W2_ac_2 = tf.placeholder(tf.float32 , shape = (dim_hidden_ac_2, dim_input_ac_2))

W1_ac_3 = tf.Variable(tf.random_uniform(shape=(dim_input_ac_3 , dim_hidden_ac_3) , minval = dmin_ac_3 , maxval = dmax_ac_3, dtype=tf.float32))
W2_ac_3 = tf.placeholder(tf.float32 , shape = (dim_hidden_ac_3, dim_input_ac_3))

W1_ac_4 = tf.Variable(tf.random_uniform(shape=(dim_input_ac_4 , dim_hidden_ac_4) , minval = dmin_ac_4 , maxval = dmax_ac_4, dtype=tf.float32))
W2_ac_4 = tf.placeholder(tf.float32 , shape = (dim_hidden_ac_4, dim_input_ac_4))

x = tf.placeholder(tf.float32 , shape=(dim_input)) # current working input vector
x2 = tf.placeholder(tf.float32 , shape=(dim_input_ac_2))
x3 = tf.placeholder(tf.float32 , shape=(dim_input_ac_3))
x4 = tf.placeholder(tf.float32 , shape=(dim_input_ac_4))

#a = tf.placeholder(tf.float32 , shape=(dim_hidden)) # current working hidden vector (pre activation)
#h = tf.placeholder(tf.float32 , shape=(dim_hidden)) # current working hidden vector (post activation)
#reonstruction = tf.placeholder(tf.float32 , shape=(dim_input)) # reconstruction of input layer MIGHT BE SUPURFLUOUS

# bias vectors
encoder_bias = tf.Variable(tf.zeros([dim_hidden], tf.float32), name="eb1")
decoder_bias = tf.Variable(tf.zeros([dim_output], tf.float32) , name="db1")

encoder_bias_ac_2 = tf.Variable(tf.zeros([dim_hidden_ac_2], tf.float32), name="eb2")
decoder_bias_ac_2 = tf.Variable(tf.zeros([dim_output_ac_2], tf.float32) , name="eb2")

encoder_bias_ac_3 = tf.Variable(tf.zeros([dim_hidden_ac_3], tf.float32), name="eb3")
decoder_bias_ac_3 = tf.Variable(tf.zeros([dim_output_ac_3], tf.float32), name="db3")

encoder_bias_ac_4 = tf.Variable(tf.zeros([dim_hidden_ac_4], tf.float32) , name="eb4")
decoder_bias_ac_4 = tf.Variable(tf.zeros([dim_output_ac_4], tf.float32) , name="db4")

# build encoder and decoder for level 1
def encoder(x):
    hidden_layer = tf.sigmoid(tf.add(tf.matmul([x] , W1) , encoder_bias))
    return hidden_layer

def decoder(h):
    reconstruction_layer = tf.add(tf.matmul(h , W2) , decoder_bias)
    return reconstruction_layer

# encoder and decoder for level 2
def encoder_ac_2(x):
    hidden_layer = tf.sigmoid(tf.add(tf.matmul([x] , W1_ac_2) , encoder_bias_ac_2))
    return hidden_layer

def decoder_ac_2(h):
    reconstruction_layer = tf.add(tf.matmul(h , W2_ac_2) , decoder_bias_ac_2)
    return reconstruction_layer

# encoder and decoder for level 3
def encoder_ac_3(x):
    hidden_layer = tf.sigmoid(tf.add(tf.matmul([x] , W1_ac_3) , encoder_bias_ac_3))
    return hidden_layer

def decoder_ac_3(h):
    reconstruction_layer = tf.add(tf.matmul(h , W2_ac_3) , decoder_bias_ac_3)
    return reconstruction_layer

# encoder and decoder for level 4
def encoder_ac_4(x):
    hidden_layer = tf.sigmoid(tf.add(tf.matmul([x] , W1_ac_4) , encoder_bias_ac_4))
    return hidden_layer

def decoder_ac_4(h):
    reconstruction_layer = tf.add(tf.matmul(h , W2_ac_4) , decoder_bias_ac_4)
    return reconstruction_layer

# Models -
encoder_operation = encoder(x)
decoder_operation = decoder(encoder_operation)

encoder_operation_ac_2 = encoder_ac_2(x2)
decoder_operation_ac_2 = decoder_ac_2(encoder_operation_ac_2)

encoder_operation_ac_3 = encoder_ac_3(x3)
decoder_operation_ac_3 = decoder_ac_3(encoder_operation_ac_3)

encoder_operation_ac_4 = encoder_ac_4(x4)
decoder_operation_ac_4 = decoder_ac_4(encoder_operation_ac_4)

# predictions
y_true = x # this is the input vectors
y_pred = decoder_operation

y_true_ac_2 = x2
y_pred_ac_2 = decoder_operation_ac_2

y_true_ac_3 = x3
y_pred_ac_3 = decoder_operation_ac_3

y_true_ac_4 = x4
y_pred_ac_4 = decoder_operation_ac_4

# loss and optimization
loss = tf.reduce_mean(tf.pow(y_pred - y_true , 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, var_list=[W1, encoder_bias, decoder_bias]) #maybe go back to gradient

loss_ac_2 = tf.reduce_mean(tf.pow(y_pred_ac_2 - y_true_ac_2 , 2))
optimizer_ac_2 = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_ac_2 , var_list=[W1_ac_2 , encoder_bias_ac_2, decoder_bias_ac_2])

loss_ac_3 = tf.reduce_mean(tf.pow(y_pred_ac_3 - y_true_ac_3 , 2))
optimizer_ac_3 = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_ac_3 , var_list=[W1_ac_3 , encoder_bias_ac_3, decoder_bias_ac_3])

loss_ac_4 = tf.reduce_mean(tf.pow(y_pred_ac_4 - y_true_ac_4 , 2))
optimizer_ac_4 = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_ac_4 , var_list=[W1_ac_4 , encoder_bias_ac_4, decoder_bias_ac_4])

# to print some results
average_loss_lvl_1 = []
average_loss_lvl_2 = []
average_loss_lvl_3 = []
average_loss_lvl_4 = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # initializes variables
    dataset = sess.run(X , {X:data})
    #print(tf.trainable_variables())
    for j in range(epoch):
        for i in range(batch_size):
            w2 = sess.run(tf.transpose(W1))
            _, l = sess.run([optimizer, loss], {x : dataset[i] , W2 : w2})
            average_loss_lvl_1.append(l)
        print("total average loss for ac_1:")
        listSum = sum(average_loss_lvl_1)
        listAvg = listSum / len(average_loss_lvl_1)
        print(listAvg)

    # training of second autoencoder using parameters of first
    for j in range(epoch):
        for i in range(batch_size):
            w2_ac_2 = sess.run(tf.transpose(W1_ac_2))
            # run our datapoint through encoder to get vector
            encoding = sess.run(encoder_operation, {x : dataset[1]})
            _, l = sess.run([optimizer_ac_2, loss_ac_2], {x2 : encoding[0] , W2_ac_2 : w2_ac_2})
            average_loss_lvl_2.append(l)

            # now perform lvl 2 encoding and decoding
        print("total average loss for ac_2:")
        listSum = sum(average_loss_lvl_2)
        listAvg = listSum / len(average_loss_lvl_2)
        print(listAvg)

    # trining of third autoencoder using parameters of second
    for j in range(epoch):
        for i in range(batch_size):
            w2_ac_3 = sess.run(tf.transpose(W1_ac_3))
            # run our datapoint through encoder to get vector
            encoding_1 = sess.run(encoder_operation, {x : dataset[1]})
            encoding_2 = sess.run(encoder_operation_ac_2 , {x2: encoding_1[0]})
            _, l = sess.run([optimizer_ac_3, loss_ac_3], {x3 : encoding_2[0] , W2_ac_3 : w2_ac_3})
            average_loss_lvl_3.append(l)

            # now perform lvl 2 encoding and decoding
        print("total average loss for ac_3:")
        listSum = sum(average_loss_lvl_3)
        listAvg = listSum / len(average_loss_lvl_3)
        print(listAvg)
    # training of fourth autoencoder parameters of third
    for j in range(epoch):
        for i in range(batch_size):
            w2_ac_4 = sess.run(tf.transpose(W1_ac_4))
            # run our datapoint through encoder to get vector
            encoding_1 = sess.run(encoder_operation, {x : dataset[1]})
            encoding_2 = sess.run(encoder_operation_ac_2 , {x2: encoding_1[0]})
            encoding_3 = sess.run(encoder_operation_ac_3 , {x3: encoding_2[0]})
            _, l = sess.run([optimizer_ac_4, loss_ac_4], {x4 : encoding_3[0] , W2_ac_4 : w2_ac_4})
            average_loss_lvl_4.append(l)

            # now perform lvl 2 encoding and decoding
        print("total average loss for ac_4:")
        listSum = sum(average_loss_lvl_4)
        listAvg = listSum / len(average_loss_lvl_4)
        print(listAvg)

    sess.close()
