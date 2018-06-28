# Helper functions controling the generation and breeding of neural networks
import random as rand

# Note on schema
# each layer is represented by BYTES_PER_LAYER, with the 1st bit of the byte string representing
# an on/off toggle, and the remaining bits representing layer capacity
# we will assume a byte will consistently be used to read the number of possible layers, as
# training a NN with more than 255 layers is infeasible

MAX_POSSIBLE_LAYERS = 7
BYTES_PER_LAYER = 2 #current max of 32,768 nodes per layer

# runction returns a random bitstring blueprint for a neural network based on defined format
def generate_blueprint():
	blueprint = ''

	# generate the random bitstring
	for i in range(MAX_POSSIBLE_LAYERS):
		num_bits = BYTES_PER_LAYER * 8
		for k in range(num_bits):
			blueprint = blueprint + str(rand.randint(0,1))

	# prepend a byte representing the number of bytes to follow (2 for every potential layer)
	num_bytes = MAX_POSSIBLE_LAYERS * BYTES_PER_LAYER
	prepend = '{0:08b}'.format(num_bytes)
	blueprint = prepend + blueprint

	return blueprint

# function that will return the first n bits of a bitstring and return the int value it represents
def read_blueprint_byte_len(blueprint):
	blueprint_len = blueprint[:8]
	return int(blueprint_len , 2)

# function that will read blueprint and return a list of integers represing the size of each layer
def read_blueprint(blueprint):
	# int list representing the structure of our neural network
	blueprint_list = [] 

	# read byte representation of every possible layer, initial values are set such that for loop sets them correctly
	# and ignores first byte (which is just a representation of max bytes in bitstring)
	start = 0
	end = 7
	for i in range(MAX_POSSIBLE_LAYERS):
		# move start and end to appropriate positions
		start = end + 1
		end = start + (8 * BYTES_PER_LAYER) - 1

		# grab our relevant bits
		layer_bits = blueprint[start:end] 

		# check to see if on/off and add them accordingly
		active = int(layer_bits[0] , 2)
		if active:
			relevant_bits = layer_bits[1:]
			blueprint_list.append(int(relevant_bits , 2))

	return blueprint_list

# function that determines whether a decoded blueprint is valid
def is_valid_structure(blueprint_structure):
	if len(blueprint_structure) > 0:
		return True
	else:
		return False


# function that accepts two bitstring blueprints, and randomly crossbreeds them, returning two new bitstrings
def breed_bitstrings(b1 , b2):
	if (len(b1) != len(b2)):
		print("CRITICAL ERROR") #do this better later

	# grab relevant substrings
	btstr1 = b1[8:]
	btstr2 = b2[8:]

	splice_point = random_splice_point(len(btstr1))

	# prepend
	num_bytes = MAX_POSSIBLE_LAYERS * BYTES_PER_LAYER
	prepend = '{0:08b}'.format(num_bytes)

	# splice two new bitstrings together
	new_btstr1 = prepend + btstr1[:splice_point] + btstr2[splice_point:]
	new_btstr2 = prepend + btstr2[:splice_point] + btstr1[splice_point:]

	return new_btstr1 , new_btstr2

	


# runction that returns a random value within range
def random_splice_point(len):
	return rand.randint(1 , len-2)
	

# function that returns random range for crossbreeding
def gen_random_range(len):
	start = len -1
	end = 0

	while (start >= end):
		start = rand.randint(0 , len-1)
		end = rand.randint(0 , len-1)

	return start , end


# function that accepts a bitstring and randomly mutates bits based on mutation_chance
def mutate_bitstring(bitstring, mutation_chance):
	# grab the bitstring minus first byte
	relevant_bits = bitstring[8:]
	num_bits = len(relevant_bits)

	for i in range(num_bits):
		# generate a random number between 0 and 1
		rand_val = rand.random()
		if custom_mutate_round(rand_val , mutation_chance):
			bit_to_flip = int(relevant_bits[i])
			if (bit_to_flip):
				bit_to_flip = 0
			else:
				bit_to_flip = 1
			relevant_bits = relevant_bits[:i] + str(bit_to_flip) + relevant_bits[(i+1):]

	# now that all potential mutations have happened prepend the first byte and return
	num_bytes = MAX_POSSIBLE_LAYERS * BYTES_PER_LAYER
	prepend = '{0:08b}'.format(num_bytes)
	return prepend + relevant_bits
		


# custom rounding function for mutating bitstrings
def custom_mutate_round(random_value_float , mutation_chance):
	return (random_value_float <= mutation_chance)






##########################################################################
# FUNCTIONS PURELY FOR DEBUGGING                                         #
##########################################################################

# prints blueprint as representative bytes
def view_blueprint_bytes(blueprint):
	start = 0
	end = 7
	for i in range(MAX_POSSIBLE_LAYERS):
		start = end + 1
		end = start + (8 * BYTES_PER_LAYER) - 1
		layer_bits = blueprint[start:end]
		display = layer_bits[:7] + " " + layer_bits[7:]
		print(display)

def random_bitstring(size):
	bitstring = ''

	for i in range(size):
		bitstring = bitstring + str(rand.randint(0,1))

	return bitstring


