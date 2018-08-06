import tensorflow as tf
import math as mat 
import numpy as np


# basic testing to make sure general structure works fundamentally
def aec_trainer_beta(epoch_size, batch_size):
	data = np.memmap('eboss.random.10000.memmap' , dtype='float32' , mode='r' , shape=(10000, 4683))

	learning_rate = 0.01

	dim_input = 4683    # the dimensionality of our dataset
	dim_hidden = 500    # number of second level nodes recommended by Van Der Maaten
	dim_output = 4683   # nodes for reconstruction vector, obviously must match input vec

	encoder_bias = tf.Variable(tf.zeros([dim_hidden], tf.float32), name="eb1")
	decoder_bias = tf.Variable(tf.zeros([dim_output], tf.float32) , name="db1")

	# we need to calculate a range for our uniform sampling
	# the recommended value is U[-b , b] where b =  sqrt(6) / sqrt(dim_layer(i) + dim_layer(i-1))
	dmax_ac_1 = mat.sqrt(6) / mat.sqrt(dim_input + 0)
	dmin_ac_1 = dmax_ac_1 * -1.0

	# placeholder for our dataset
	X = tf.placeholder(tf.float32 , shape=(None, dim_input))
	x = tf.placeholder(tf.float32 , shape=(dim_input)) # current working input vector

	W1 = tf.Variable(tf.random_uniform(shape=(dim_input , dim_hidden) , minval = dmin_ac_1 , maxval = dmax_ac_1, dtype=tf.float32))
	W2 = tf.placeholder(tf.float32 , shape = (dim_hidden, dim_input))

	# build encoder and decoder for level 1
	def encoder(x):
	    hidden_layer = tf.sigmoid(tf.add(tf.matmul([x] , W1) , encoder_bias))
	    return hidden_layer

	def decoder(h):
	    reconstruction_layer = tf.add(tf.matmul(h , W2) , decoder_bias)
	    return reconstruction_layer

	encoder_operation = encoder(x)
	decoder_operation = decoder(encoder_operation)

	y_true = x # this is the input vectors
	y_pred = decoder_operation

	loss = tf.reduce_mean(tf.pow(y_pred - y_true , 2))
	optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, var_list=[W1, encoder_bias, decoder_bias]) #maybe go back to gradient

	average_loss_lvl_1 = []

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer()) # populates our globals
		dataset = sess.run(X, {X:data})

		for j in range(epoch_size):
			for i in range(batch_size):
				w2 = sess.run(tf.transpose(W1))
				_, l = sess.run([optimizer, loss], {x : dataset[i] , W2 : w2})
				average_loss_lvl_1.append(l)
			print("total average loss for ac_1:")
			listSum = sum(average_loss_lvl_1)
			listAvg = listSum / len(average_loss_lvl_1)
			print(listAvg)
		sess.close()


def test_beta_func():
	aec_trainer_beta(10, 100)