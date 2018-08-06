# Overall driver to test multiple generations of a bred neural network, seeking to continually
# reduce overall error in parametric tSNE

# necessary package imports
import tensorflow as tf 
import genetic_helpers as gh 


# parameters that can be set to tweak overall proceadure
NUM_GENERATIONS = 1
GENERATION_SIZE = 1

# list containing our current generation
generation_members = []


# iterate through given number of generations
for g in range(NUM_GENERATIONS):
	pass # PLACEHOLDER
	# iterate through generation_members and train each network
		# record the id and performance of each network
	# write out documentation for generation's info and performance
	# take top X performers
	# breed new generation

# profit
# but also send off to graph somehow