# class definition of an individual network for training t-sne
# class can be fed a string of predefined structure and will generate a
# structure for the neural network to be trained
import tensorflow as tf 
import numpy as np 
import genetic_helpers as gh 

class Bred_PTSNE_NN:
	def __init__(self, name):
		self.name = name
	layers_size_dict = {}
	weights_dict = {}
	weights = []
	bias_dict = {}
	num_layers = 0

	# function that allows layers to dynamically be added to the neural network
	def add_layer(self, cardinality):
		self.num_layers = self.num_layers + 1
		layer_name = "layer_" +str(self.num_layers)
		self.layers_size_dict[layer_name] = cardinality

	# function that prepares the neural network for training
	def setup_structure():
		pass

	# function that trains the neural network on given data with given epochs
	def train_neural_network(dataset , epochs):
		pass


	# for testing
	def print_structure(self):
		print(self.layers_size_dict)




# function will generate an object of class Bred_PTSNE_NN based on a given name and
# dna sequence
# the generation of the NN, itself a stack of autoencoders will be of the following form
# the first 8 bits of the string will be ignored (it is simply a specification of size of the dna string
# in bytes). This function will then read num_bytes_per_layer bytes in preparation for adding a layer
# to the generated nn. The first bit will specify whether the layer is to be read at all, and the 
# subsequent bits will be given to the instance of the class to generate a layer of that size
def generate_nn(name , dna_string):
	bit_pointer = 8
	num_possible_layers = dna_string[:bit_pointer] # tells us how many times to iterate

	# instantiate our neural network
	construct = Bred_PTSNE_NN(name)

	# get int representation of dna string
	structure = gh.read_blueprint(dna_string)

	for vals in structure:
		construct.add_layer(vals)
	
	return construct




		
