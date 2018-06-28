# tests for genetic helpers
import genetic_helpers as g
import random as rand

def test_generations():
	generations = 50

	num_invalid_blueprints = 0

	list_of_layer_sizes = []
	layer_sizes_greater_than_4 = []

	for i in range(generations):
		blueprint = g.generate_blueprint()
		structure = g.read_blueprint(blueprint)
		if not g.is_valid_structure(structure):
			num_invalid_blueprints = num_invalid_blueprints + 1
		else:
			list_of_layer_sizes.append(len(structure))
			if(len(structure) > 4):
				layer_sizes_greater_than_4.append(len(structure))

	print("We generated ", str(num_invalid_blueprints) , " invalid structures")
	print("We generated " , str(len(list_of_layer_sizes)) , " valid structures")
	print("Of the valid structures, the average layer size was:" , str(sum(list_of_layer_sizes) / len(list_of_layer_sizes)))
	print("Of the valid structures, there were " , str(len(layer_sizes_greater_than_4)) , " with 5-7")

def test_mutations():
	for i in range(100):
		# make random bitstring
		btstr = g.random_bitstring(16)
		m_btstr = g.mutate_bitstring(btstr , .1)
		print("our bitstring is  : " , btstr)
		print("our m bitstring is: " , m_btstr)
		print(" ")

def test_crossbreeding():
	btstr1 = g.random_bitstring(24)
	btstr2 = g.random_bitstring(24)

	x, y = g.breed_bitstrings(btstr1 , btstr2)

	print("Bitstring  1: " , btstr1)
	print("Bitstring  2: " , btstr2)
	print("Crossbreed 1: ",  x)
	print("Crossbreed 2: ",  y)