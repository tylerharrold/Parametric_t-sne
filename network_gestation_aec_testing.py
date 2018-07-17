import network_gestation_aec as nga 
import genetic_helpers as gh 

blueprint = gh.generate_blueprint()

my_network = nga.generate_nn("test" , blueprint)

my_network.print_structure()

