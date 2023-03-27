import random
import time
import numpy as np
from console.utils import cls
import matplotlib.pyplot as plt
from pyplus import alpha

def imprint():
    pass

class Params:
    def __init__(self):
        # This acts as a configuration object that holds various settings and parameters for the simulation
        self.minGenomeLength = 8 # Minimum intelligence level for the neural network
        self.maxGenomeLength = 8 # Maximum intelligence level for the neural network
        self.maxAge = 80 # Doesn't do anything
        self.grid_height = 70 # width
        self.grid_width = 30 # height
        self.grid_size = round((self.grid_height * self.grid_width) / ((self.grid_height + self.grid_width) / 2)) # Don't change
        self.speed = 1 # How much creatures move each time 
        self.vision = 1
        self.biodiversityCount = 3 # How many species total
        self.creatureProportion = 5 # How many creatures are created depending on the grid_size
        self.minCreatures = 3 # The minimum value of any population until the simulation resets to the next generation
        self.mutationRate = 2 # How much a creature can mutate while reproducing
        self.mirror = False # Due to the borders of the grid using modulo, mirror prints the grid twice, seamlessly
        self.reproducing_hunger = 5 # Minimum food until you can't reproduce
        self.reproductionProportion = 1 # How much food a creature loses in proportion to their hunger
        self.fullness = 1
        self.pheromones = 3

p = Params()

def cdef(func,fdef):
    def wrapper(arg=fdef):
        ret = func(arg)
        return ret
    return wrapper

rst = '\033[0m'
def generate_display():
    # list of uppercase alphabet characters
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    # list of ANSI escape sequence colors (roy g biv)
    colors = []
    for i in range(0, 16):
        for j in range(0, 16):
            code = str(i * 16 + j)
            colors.append(u"\u001b[38;5;" + code + "m")

    # list to store selected characters and colors
    fselected = []

    # loop to select randok characters and colors
    for i in alpha(p.biodiversityCount):
        selected_char = random.choice(alphabet)
        alphabet.remove(selected_char)
        selected_color = random.choice(colors)
        colors.remove(selected_color)
        fselected.append(selected_color + selected_char)
    return fselected

class Gene:
    def __init__(self,): # Sink number represents the actions a gene could take
        self.sourceNum = random.randint(0,3)
        self.sinkNum = random.randint(0,(10+p.pheromones))
    def makeCustomGene(self, source, sink, weight):
        self.sourceNum = source
        self.sinkNum = sink
        self.weight = weight
        
def makeRandomGene():
    return Gene()
    
class Genome:
    def __init__(self):
        self.genes = []

    def add_gene(self, gene):
        self.genes.append(gene)
        
    def mutate(self):
        for i in alpha(random.randrange(p.mutationRate)): # It loops a random number of times based on mutationRate
            if random.random() <= 0.5:
                mutated_gene = random.randrange(len(self.genes))
                self.genes[mutated_gene].weight = random.uniform(-4.0, 4.0) # It randomly selects a gene in the genes list sets the weight to a random value
        return self
        

def makeRandomGenome(): # Creates a genome, and appends the Gene class to the genome: 'genes' list.
    genome = Genome()
    length = random.randint(p.minGenomeLength, p.maxGenomeLength)
    for i in alpha(length):
        genome.add_gene(makeRandomGene())
    return genome

class NeuralNetwork:
    
    def create_network(self):
        # initialize weights and biases
        for i in alpha(len(self.genome.genes)):
            self.weights.append(random.uniform(-4, 4))
            self.biases.append(random.uniform(-1, 1))
        
        # reshape weights and biases into matrices
        self.weights = np.reshape(self.weights, (len(self.weights), 1))
        self.biases = np.reshape(self.biases, (len(self.biases), 1))

    def predict_feedforward(self, input_data):
        # calculate the dot product of the input data and weights
        self.input_layer = np.dot(self.weights, input_data)
        
        # add the biases to the input layer
        self.input_layer = np.add(self.input_layer, self.biases)
        
        # pass the input layer through a activation function
        self.hidden_layer = np.maximum(0, self.input_layer)
        
        # return the output of the network
        self.output_layer = self.hidden_layer
        return self.output_layer

    def __init__(self, genome):
        self.genome = genome
        self.weights = []
        self.biases = []
        
        self.input_layer = []
        self.hidden_layer = []
        self.output_layer = []
        
        self.create_network()

# Species
class Life:


    # Individual
    class Creature:
            
        # Find the distance from an alien
        def DETECTION_ALIEN_RADIUS(self):
            distances = []

            # Skip if the same species
            for species in species_list:
                if species != self.species:

                    # Calculate the the grid size minus the absolute value of the difference between distances divided by the grid size
                    for creature in species.creature_list:
                        x = (p.grid_width - abs(self.x - creature.x)) / p.grid_width
                        y = (p.grid_height - abs(self.y - creature.y)) / p.grid_height
                        distances.append((x + y)/2)
                    
            # Sort distance values by biggest to smallest (closest to farthest)
            distances.sort(reverse=True)
            
            # Return the largest distance value if other species exist, else return 0
            try:
                return distances[0]
            except:
                return 0


        # Find the distance from family
        def DETECTION_FAMILY_RADIUS(self):
            distances = []
            
            # Calculate the the grid size minus the absolute value of the difference between distances divided by the grid size
            for creature in self.species.creature_list:
                if creature != self:
                    x = (p.grid_width - abs(self.x - creature.x)) / p.grid_width
                    y = (p.grid_height - abs(self.y - creature.y)) / p.grid_height
                    distances.append((x + y)/2)
                
            # Sort distance values by biggest to smallest (closest to farthest)
            distances.sort(reverse=True)
            
            # Return the largest distance value if other species exist, else return 0
            try:
                return distances[0]
            except:
                return 0
        
        def FAMILY_NICHE(self):
            sorted_list = sorted(self.species.creature_list, key=lambda creature: creature.hunger, reverse=True)
            for creature in sorted_list:
                if abs(self.x - creature.x) <= self.vision and abs(self.y - creature.y) <= self.vision:
                    creature.hunger -= 1
                    self.hunger += 2
                return True



        # Duplicates and mutates an offspring if the following requirements are met
        def REPRODUCE(self):
            if self.hunger < p.reproducing_hunger and self.age > 0 and self.health > 0:
                self.species.creature_list.append(
                    self.species.Creature(
                        (self.x + random.randint(-1, 1))%p.grid_width, 
                        (self.y + random.randint(-1, 1))%p.grid_height, 
                        self.speed, self.vision,
                        self.genome.mutate(), 
                        self.direction, 
                        self.enemy_index,
                        self.species
                    )
                )
                self.hunger += p.reproducing_hunger / p.reproductionProportion


        # This is how the creatures eat and get food
        def KILL_FORWARD(self):
            for creature in species_list[self.enemy_index].creature_list:
                if abs(self.x - creature.x) <= self.vision and abs(self.y - creature.y) <= self.vision and self.hunger > p.fullness:
                    grid[creature.rx][creature.ry] = " "
                    creature.display = " "
                    species_list[self.enemy_index].creature_list.remove(creature)
                    self.hunger -= 10
                    self.foodHighscore += 1
                    self.health += 5
                    if self.health > 100: self.health = 100
                    return True
            return False
            
            
            
        #  Randomly select an input from 1 to 0
        def RANDOM(self):
            return random.random()
            
        # Make an input depending on how extinct their food is
        def ECO_NICHE(self):
            try:
                return (p.grid_size/5) / len(species_list[self.enemy_index].creature_list) 
            except ZeroDivisionError: 
                return 0
                
        def FAMILY_DENSITY(self):
            nearby_family = len([creature for creature in self.species.creature_list if abs(self.x - creature.x) <= self.vision + 1 and abs(self.y - creature.y) <= self.vision + 1])
            nearby_alien = len([creature for creature in species_list[self.enemy_index].creature_list if abs(self.x - creature.x) <= self.vision + 1 and abs(self.y - creature.y) <= self.vision + 1])
            total = nearby_family + nearby_alien
            try:
                return nearby_family / total
            except ZeroDivisionError:
                return 0
            
        def ALIEN_DENSITY(self):
            nearby_family = len([creature for creature in self.species.creature_list if abs(self.x - creature.x) <= self.vision + 1 and abs(self.y - creature.y) <= self.vision + 1])
            nearby_alien = len([creature for creature in species_list[self.enemy_index].creature_list if abs(self.x - creature.x) <= self.vision + 1 and abs(self.y - creature.y) <= self.vision + 1])
            total = nearby_family + nearby_alien
            try:
                return nearby_alien / total
            except ZeroDivisionError:
                return 0
                
        def PHEROMONE_DENSITY(self, ptype):
            nearby_pheromones = len([pheromone for pheromone in ptype.pheromone_list if abs(self.x - pheromone.x) <= self.vision + 1 and abs(self.y - pheromone.y) <= self.vision + 1])
            nearby_family = len([creature for creature in self.species.creature_list if abs(self.x - creature.x) <= self.vision + 1 and abs(self.y - creature.y) <= self.vision + 1])
            total = nearby_family + nearby_pheromones
            try:
                return nearby_pheromones / total
            except ZeroDivisionError:
                return 0
            
        def MOVE_FORWARD(self):
            self.x = (self.x + (self.speed * np.cos(theta(self.direction))))%p.grid_width
            self.y = (self.y + (self.speed * np.sin(theta(self.direction))))%p.grid_height
            self.rx = round(self.x)
            self.ry = round(self.y)
        def MOVE_LEFT(self):
            self.x = (self.x + (self.speed * np.cos(theta(self.direction-90))))%p.grid_width
            self.y = (self.y + (self.speed * np.sin(theta(self.direction-90))))%p.grid_height
            self.rx = round(self.x)
            self.ry = round(self.y)
        def MOVE_RIGHT(self):
            self.x = (self.x + (self.speed * np.cos(theta(self.direction+90))))%p.grid_width
            self.y = (self.y + (self.speed * np.sin(theta(self.direction+90))))%p.grid_height
            self.rx = round(self.x)
            self.ry = round(self.y)
        def MOVE_BACK(self):
            self.x = (self.x + (self.speed * np.cos(theta(self.direction+180))))%p.grid_width
            self.y = (self.y + (self.speed * np.sin(theta(self.direction+180))))%p.grid_height
            self.rx = round(self.x)
            self.ry = round(self.y)
        def TURN_LEFT(self):
            self.direction -= 15
        def TURN_RIGHT(self):
            self.direction += 15
        def TURN_BACK(self):
            self.direction += 180
        def TURN_RANDOM(self):
            self.direction = random.randint(0,359)
        def LOG_COORDS(self):
            self.homecoords = (self.x, self.y)
        def FIND_ENEMY(self):
            sorted_list = sorted(species_list[self.enemy_index].creature_list, key=lambda c: distance(difference(self.x, c.x), difference(self.y, c.y)), reverse=True)
            self.enemycoords = (sorted_list[0].x, sorted_list[0].y)
        def GO_ENEMY(self):
            x1, y1 = self.x, self.y
            x2, y2 = self.enemycoords
            xdis = difference(x1, x2)
            ydis = difference(y1, y2)
            radius = distance(xdis, ydis)
            self.x = (self.x + (xdis/radius))%p.grid_width
            self.y = (self.y + (ydis/radius))%p.grid_height
            self.rx = round(self.x)
            self.ry = round(self.y)
        def GO_HOME(self):
            x1, y1 = self.x, self.y
            x2, y2 = self.homecoords
            xdis = difference(x1, x2)
            ydis = difference(y1, y2)
            radius = distance(xdis, ydis)
            self.x = (self.x + (xdis/radius))%p.grid_width
            self.y = (self.y + (ydis/radius))%p.grid_height
            self.rx = round(self.x)
            self.ry = round(self.y)
        def SECRETE_PHEROMONE(self, ptype):
            ptype.pheromone_list.append(ptype.Pheromone(self.x, self.y, ptype))
        # Initialize the creature:
        def __init__(self, x: int, y: int, speed: int, vision: int, genome: Genome, direction: int, enemy_index: int, species):
            self.x = x
            self.y = y
            self.speed = speed
            self.vision = vision
            self.hunger = 1
            self.age = 0
            self.health = 100
            self.genome = genome
            self.last_dir = direction
            self.direction = direction
            self.enemy_index = enemy_index
            self.species = species
            self.display = selected[self.enemy_index + 1]
            self.foodHighscore = 0
            self.homecoords = (self.x, self.y)
            self.enemycoords = (self.x, self.y)
            self.rx = round(self.x)
            self.ry = round(self.y)
            
            # Initialize the neural network with the weights and biases from the genome
            self.neural_network = NeuralNetwork(self.genome)

            self.Sensors = [
                self.DETECTION_ALIEN_RADIUS,
                self.DETECTION_FAMILY_RADIUS,
                self.RANDOM,
                self.ECO_NICHE,
                self.FAMILY_DENSITY, 
                self.ALIEN_DENSITY
            ]
            self.Sensors.extend([cdef(self.PHEROMONE_DENSITY,ptype) for ptype in ptypes])
            
            self.Actions = [
                self.FAMILY_NICHE,
                self.MOVE_FORWARD,
                self.MOVE_LEFT,
                self.MOVE_RIGHT,
                self.MOVE_BACK,
                self.TURN_LEFT,
                self.TURN_RIGHT,
                self.TURN_BACK,
                self.TURN_RANDOM,
                self.FIND_ENEMY,
                self.GO_ENEMY
            ]
            self.Actions.extend([cdef(self.SECRETE_PHEROMONE,ptype) for ptype in ptypes])


        # Use the functions and environment to decide what to do
        def executeActions(self):
            def possible_actions(self) -> list: # Create a list of possible actions a single creature can take depending on the sinkNums in their genome.
                Actions_list = []
                for gene in self.genome.genes:
                    Actions_list.append(self.Actions[gene.sinkNum])
                return Actions_list
            a = possible_actions(self)

            # Get the input values from the sensors
            input_data = [[func() for func in self.Sensors]*len(self.genome.genes)]

            # Pass the input values through the neural network to get the output values
            outputs = self.neural_network.predict_feedforward(input_data).argmax(axis=1)

            # Calculate the probabilities using the softmax function
            probabilities = np.exp(outputs) / np.sum(np.exp(outputs))

            # Choose an action based on the highest probabilities
            action_index = np.random.choice(range(len(a)), p=probabilities)

            # Perform the selected action
            a[action_index]()
            self.hunger += 1
            self.age += 1
            
            self.KILL_FORWARD()
            
            self.REPRODUCE()



    def findSpecies(self):
        for i, species in enumerate(species_list):
            if species == self:
                return i, species


    def __init__(self):
        self.creature_list = []
        self.values = []


    def create_population(self, size):
        self.creature_list = [
            self.Creature(
                random.randint(0, p.grid_width), 
                random.randint(0, p.grid_height), 
                p.speed, 
                p.vision, 
                best_creatures[self.findSpecies()[0]].genome.mutate() if best_creatures != 0 else makeRandomGenome(), 
                random.randrange(4),
                self.findSpecies()[0] - 1,
                self.findSpecies()[1]
                ) 
            for i in alpha(size)
        ]

class Pheromone_type:
    class Pheromone:
        def __init__(self, x, y, ptype):
            self.x = x
            self.y = y
            self.ptype = ptype
            
        def DISSIPATE(self):
            self.x = (self.x + random.uniform(-0.1, 0.1))%p.grid_width
            self.y = (self.y + random.uniform(-0.1, 0.1))%p.grid_height
            if random.random() < 0.1:
                pass
    def __init__(self):
        self.pheromone_list = []
        
def fitness_function():
    foodHighscores = []
    bestCreatures = []
    for i,species in enumerate(species_list):
        bestCreatures.append(0)
        foodHighscores.append(-1)
        for creature in species_list[i].creature_list:
            if creature.foodHighscore > foodHighscores[i]: bestCreatures[i], foodHighscores[i] = creature, creature.foodHighscore
    return bestCreatures
def difference(a, b):
    return a - b
def theta(degrees):
    return degrees * (np.pi/180)
def distance(a, b):
    return np.sqrt(a**2 + b**2) + 0.0000001
def go(lists):
    # Move and eat for creatures
    j = 0
    # Iterates a while loop for every creature in the simulation
    while j < len(lists):
        grid[lists[j].rx][lists[j].ry] = ' '
        lists[j].executeActions()
        # Repeats the loop if a creature dies so the index (j) doesn't go out of range
        if lists[j].hunger >= 30 or lists[j].age > p.maxAge+random.randint(0,20) or lists[j].health < 0:
            lists.pop(j)
        else:
            #put the prey on the grid
            if lists[j].age > (p.maxAge / 4.44444444444):
                grid[lists[j].rx][lists[j].ry] = lists[j].display
                
            else:
                grid[lists[j].rx][lists[j].ry] = str.lower(lists[j].display)
            j += 1


def popu(species):
    return sum(list(map(lambda x: len(x.creature_list), species)))


# Functions for every 'frame' in the simulation
def simulate():
    
    # Activate each species in the species list and clear the previous grid
    for species in species_list:
        go(species.creature_list)
    cls() # Clears screen

    # Create the population variable and print the total populations of each species
    population = 0
    for i,species in enumerate(species_list):
        print(f"{selected[i]}: {len(species.creature_list)}")

    # Print outputs for better understanding]
    print(f"{rst}Population: {popu(species_list)}")
    print(f"Current gen: {gen}")
    print(f"Current sim: {simulationLength}")
    print(f"Longest sim: {simulationLengthHighest}")
    
    # Use mirrored mode if enabled
    if p.mirror == True:
        for row in grid:
            print(*row, sep =' ', end=" ", flush=True)
            print(*row, sep =' ', end=" ", flush=True)
            print()
        return

    # Print grid values
    for row in reversed(grid): print(*row, sep=" ", flush=True)


# Define game variables
simulationLengthHighest = 0
best_creatures = 0
selected = generate_display()
gen = 0

# Main game loop
while True:

    # Define generational variables
    grid_sp = 1
    grid = [[' ' for _ in alpha(p.grid_height+2)] for _ in alpha(p.grid_width+2)]
    simulationLength = 0
    species_list = []
    ptypes = []
    length = []
    gen += 1
    
    for i in alpha(p.pheromones):
        ptypes.append(Pheromone_type())
    
    # Create a species and population for each 
    for i in range(p.biodiversityCount):
        species_list.append(Life())
        species_list[i].create_population(round(p.grid_size/(p.biodiversityCount / p.creatureProportion)))
    
    # Simulation loop for every frame
    breakloop = False
    while True:
        
        # Detect when to reset and go to the next generation
        for species in species_list:
            if len(species.creature_list) < p.minCreatures:
                best_creatures = fitness_function()
                print("Fail!")
                breakloop = True
        if breakloop: break

        # Simulate a frame and increase the simulationLength/time for the pyplot
        simulate()
        simulationLength += 1
        length.append(simulationLength)
        for species in species_list:
            species.values.append(len(species.creature_list))
            plt.plot(length,species.values)
        plt.draw()
        plt.pause(0.01)
        #plt.clf()
        if simulationLength > simulationLengthHighest: simulationLengthHighest = simulationLength
        time.sleep(0.02) # Allows the user to see the simulation images 1 by 1import random
import time
import numpy as np
from console.utils import cls
import matplotlib.pyplot as plt
from pyplus import alpha

def imprint():
    pass

class Params:
    def __init__(self):
        # This acts as a configuration object that holds various settings and parameters for the simulation
        self.minGenomeLength = 8 # Minimum intelligence level for the neural network
        self.maxGenomeLength = 8 # Maximum intelligence level for the neural network
        self.maxAge = 80 # Doesn't do anything
        self.grid_height = 70 # width
        self.grid_width = 30 # height
        self.grid_size = round((self.grid_height * self.grid_width) / ((self.grid_height + self.grid_width) / 2)) # Don't change
        self.speed = 1 # How much creatures move each time 
        self.vision = 1
        self.biodiversityCount = 3 # How many species total
        self.creatureProportion = 5 # How many creatures are created depending on the grid_size
        self.minCreatures = 3 # The minimum value of any population until the simulation resets to the next generation
        self.mutationRate = 2 # How much a creature can mutate while reproducing
        self.mirror = False # Due to the borders of the grid using modulo, mirror prints the grid twice, seamlessly
        self.reproducing_hunger = 5 # Minimum food until you can't reproduce
        self.reproductionProportion = 1 # How much food a creature loses in proportion to their hunger
        self.fullness = 1
        self.pheromones = 3

p = Params()

def cdef(func,fdef):
    def wrapper(arg=fdef):
        ret = func(arg)
        return ret
    return wrapper

rst = '\033[0m'
def generate_display():
    # list of uppercase alphabet characters
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    # list of ANSI escape sequence colors (roy g biv)
    colors = []
    for i in range(0, 16):
        for j in range(0, 16):
            code = str(i * 16 + j)
            colors.append(u"\u001b[38;5;" + code + "m")

    # list to store selected characters and colors
    fselected = []

    # loop to select randok characters and colors
    for i in alpha(p.biodiversityCount):
        selected_char = random.choice(alphabet)
        alphabet.remove(selected_char)
        selected_color = random.choice(colors)
        colors.remove(selected_color)
        fselected.append(selected_color + selected_char)
    return fselected

class Gene:
    def __init__(self,): # Sink number represents the actions a gene could take
        self.sourceNum = random.randint(0,3)
        self.sinkNum = random.randint(0,(10+p.pheromones))
    def makeCustomGene(self, source, sink, weight):
        self.sourceNum = source
        self.sinkNum = sink
        self.weight = weight
        
def makeRandomGene():
    return Gene()
    
class Genome:
    def __init__(self):
        self.genes = []

    def add_gene(self, gene):
        self.genes.append(gene)
        
    def mutate(self):
        for i in alpha(random.randrange(p.mutationRate)): # It loops a random number of times based on mutationRate
            if random.random() <= 0.5:
                mutated_gene = random.randrange(len(self.genes))
                self.genes[mutated_gene].weight = random.uniform(-4.0, 4.0) # It randomly selects a gene in the genes list sets the weight to a random value
        return self
        

def makeRandomGenome(): # Creates a genome, and appends the Gene class to the genome: 'genes' list.
    genome = Genome()
    length = random.randint(p.minGenomeLength, p.maxGenomeLength)
    for i in alpha(length):
        genome.add_gene(makeRandomGene())
    return genome

class NeuralNetwork:
    
    def create_network(self):
        # initialize weights and biases
        for i in alpha(len(self.genome.genes)):
            self.weights.append(random.uniform(-4, 4))
            self.biases.append(random.uniform(-1, 1))
        
        # reshape weights and biases into matrices
        self.weights = np.reshape(self.weights, (len(self.weights), 1))
        self.biases = np.reshape(self.biases, (len(self.biases), 1))

    def predict_feedforward(self, input_data):
        # calculate the dot product of the input data and weights
        self.input_layer = np.dot(self.weights, input_data)
        
        # add the biases to the input layer
        self.input_layer = np.add(self.input_layer, self.biases)
        
        # pass the input layer through a activation function
        self.hidden_layer = np.maximum(0, self.input_layer)
        
        # return the output of the network
        self.output_layer = self.hidden_layer
        return self.output_layer

    def __init__(self, genome):
        self.genome = genome
        self.weights = []
        self.biases = []
        
        self.input_layer = []
        self.hidden_layer = []
        self.output_layer = []
        
        self.create_network()

# Species
class Life:


    # Individual
    class Creature:
            
        # Find the distance from an alien
        def DETECTION_ALIEN_RADIUS(self):
            distances = []

            # Skip if the same species
            for species in species_list:
                if species != self.species:

                    # Calculate the the grid size minus the absolute value of the difference between distances divided by the grid size
                    for creature in species.creature_list:
                        x = (p.grid_width - abs(self.x - creature.x)) / p.grid_width
                        y = (p.grid_height - abs(self.y - creature.y)) / p.grid_height
                        distances.append((x + y)/2)
                    
            # Sort distance values by biggest to smallest (closest to farthest)
            distances.sort(reverse=True)
            
            # Return the largest distance value if other species exist, else return 0
            try:
                return distances[0]
            except:
                return 0


        # Find the distance from family
        def DETECTION_FAMILY_RADIUS(self):
            distances = []
            
            # Calculate the the grid size minus the absolute value of the difference between distances divided by the grid size
            for creature in self.species.creature_list:
                if creature != self:
                    x = (p.grid_width - abs(self.x - creature.x)) / p.grid_width
                    y = (p.grid_height - abs(self.y - creature.y)) / p.grid_height
                    distances.append((x + y)/2)
                
            # Sort distance values by biggest to smallest (closest to farthest)
            distances.sort(reverse=True)
            
            # Return the largest distance value if other species exist, else return 0
            try:
                return distances[0]
            except:
                return 0
        
        def FAMILY_NICHE(self):
            sorted_list = sorted(self.species.creature_list, key=lambda creature: creature.hunger, reverse=True)
            for creature in sorted_list:
                if abs(self.x - creature.x) <= self.vision and abs(self.y - creature.y) <= self.vision:
                    creature.hunger -= 1
                    self.hunger += 2
                return True



        # Duplicates and mutates an offspring if the following requirements are met
        def REPRODUCE(self):
            if self.hunger < p.reproducing_hunger and self.age > 0 and self.health > 0:
                self.species.creature_list.append(
                    self.species.Creature(
                        (self.x + random.randint(-1, 1))%p.grid_width, 
                        (self.y + random.randint(-1, 1))%p.grid_height, 
                        self.speed, self.vision,
                        self.genome.mutate(), 
                        self.direction, 
                        self.enemy_index,
                        self.species
                    )
                )
                self.hunger += p.reproducing_hunger / p.reproductionProportion


        # This is how the creatures eat and get food
        def KILL_FORWARD(self):
            for creature in species_list[self.enemy_index].creature_list:
                if abs(self.x - creature.x) <= self.vision and abs(self.y - creature.y) <= self.vision and self.hunger > p.fullness:
                    grid[creature.rx][creature.ry] = " "
                    creature.display = " "
                    species_list[self.enemy_index].creature_list.remove(creature)
                    self.hunger -= 10
                    self.foodHighscore += 1
                    self.health += 5
                    if self.health > 100: self.health = 100
                    return True
            return False
            
            
            
        #  Randomly select an input from 1 to 0
        def RANDOM(self):
            return random.random()
            
        # Make an input depending on how extinct their food is
        def ECO_NICHE(self):
            try:
                return (p.grid_size/5) / len(species_list[self.enemy_index].creature_list) 
            except ZeroDivisionError: 
                return 0
                
        def FAMILY_DENSITY(self):
            nearby_family = len([creature for creature in self.species.creature_list if abs(self.x - creature.x) <= self.vision + 1 and abs(self.y - creature.y) <= self.vision + 1])
            nearby_alien = len([creature for creature in species_list[self.enemy_index].creature_list if abs(self.x - creature.x) <= self.vision + 1 and abs(self.y - creature.y) <= self.vision + 1])
            total = nearby_family + nearby_alien
            try:
                return nearby_family / total
            except ZeroDivisionError:
                return 0
            
        def ALIEN_DENSITY(self):
            nearby_family = len([creature for creature in self.species.creature_list if abs(self.x - creature.x) <= self.vision + 1 and abs(self.y - creature.y) <= self.vision + 1])
            nearby_alien = len([creature for creature in species_list[self.enemy_index].creature_list if abs(self.x - creature.x) <= self.vision + 1 and abs(self.y - creature.y) <= self.vision + 1])
            total = nearby_family + nearby_alien
            try:
                return nearby_alien / total
            except ZeroDivisionError:
                return 0
                
        def PHEROMONE_DENSITY(self, ptype):
            nearby_pheromones = len([pheromone for pheromone in ptype.pheromone_list if abs(self.x - pheromone.x) <= self.vision + 1 and abs(self.y - pheromone.y) <= self.vision + 1])
            nearby_family = len([creature for creature in self.species.creature_list if abs(self.x - creature.x) <= self.vision + 1 and abs(self.y - creature.y) <= self.vision + 1])
            total = nearby_family + nearby_pheromones
            try:
                return nearby_pheromones / total
            except ZeroDivisionError:
                return 0
            
        def MOVE_FORWARD(self):
            self.x = (self.x + (self.speed * np.cos(theta(self.direction))))%p.grid_width
            self.y = (self.y + (self.speed * np.sin(theta(self.direction))))%p.grid_height
            self.rx = round(self.x)
            self.ry = round(self.y)
        def MOVE_LEFT(self):
            self.x = (self.x + (self.speed * np.cos(theta(self.direction-90))))%p.grid_width
            self.y = (self.y + (self.speed * np.sin(theta(self.direction-90))))%p.grid_height
            self.rx = round(self.x)
            self.ry = round(self.y)
        def MOVE_RIGHT(self):
            self.x = (self.x + (self.speed * np.cos(theta(self.direction+90))))%p.grid_width
            self.y = (self.y + (self.speed * np.sin(theta(self.direction+90))))%p.grid_height
            self.rx = round(self.x)
            self.ry = round(self.y)
        def MOVE_BACK(self):
            self.x = (self.x + (self.speed * np.cos(theta(self.direction+180))))%p.grid_width
            self.y = (self.y + (self.speed * np.sin(theta(self.direction+180))))%p.grid_height
            self.rx = round(self.x)
            self.ry = round(self.y)
        def TURN_LEFT(self):
            self.direction -= 15
        def TURN_RIGHT(self):
            self.direction += 15
        def TURN_BACK(self):
            self.direction += 180
        def TURN_RANDOM(self):
            self.direction = random.randint(0,359)
        def LOG_COORDS(self):
            self.homecoords = (self.x, self.y)
        def FIND_ENEMY(self):
            sorted_list = sorted(species_list[self.enemy_index].creature_list, key=lambda c: distance(difference(self.x, c.x), difference(self.y, c.y)), reverse=True)
            self.enemycoords = (sorted_list[0].x, sorted_list[0].y)
        def GO_ENEMY(self):
            x1, y1 = self.x, self.y
            x2, y2 = self.enemycoords
            xdis = difference(x1, x2)
            ydis = difference(y1, y2)
            radius = distance(xdis, ydis)
            self.x = (self.x + (xdis/radius))%p.grid_width
            self.y = (self.y + (ydis/radius))%p.grid_height
            self.rx = round(self.x)
            self.ry = round(self.y)
        def GO_HOME(self):
            x1, y1 = self.x, self.y
            x2, y2 = self.homecoords
            xdis = difference(x1, x2)
            ydis = difference(y1, y2)
            radius = distance(xdis, ydis)
            self.x = (self.x + (xdis/radius))%p.grid_width
            self.y = (self.y + (ydis/radius))%p.grid_height
            self.rx = round(self.x)
            self.ry = round(self.y)
        def SECRETE_PHEROMONE(self, ptype):
            ptype.pheromone_list.append(ptype.Pheromone(self.x, self.y, ptype))
        # Initialize the creature:
        def __init__(self, x: int, y: int, speed: int, vision: int, genome: Genome, direction: int, enemy_index: int, species):
            self.x = x
            self.y = y
            self.speed = speed
            self.vision = vision
            self.hunger = 1
            self.age = 0
            self.health = 100
            self.genome = genome
            self.last_dir = direction
            self.direction = direction
            self.enemy_index = enemy_index
            self.species = species
            self.display = selected[self.enemy_index + 1]
            self.foodHighscore = 0
            self.homecoords = (self.x, self.y)
            self.enemycoords = (self.x, self.y)
            self.rx = round(self.x)
            self.ry = round(self.y)
            
            # Initialize the neural network with the weights and biases from the genome
            self.neural_network = NeuralNetwork(self.genome)

            self.Sensors = [
                self.DETECTION_ALIEN_RADIUS,
                self.DETECTION_FAMILY_RADIUS,
                self.RANDOM,
                self.ECO_NICHE,
                self.FAMILY_DENSITY, 
                self.ALIEN_DENSITY
            ]
            self.Sensors.extend([cdef(self.PHEROMONE_DENSITY,ptype) for ptype in ptypes])
            
            self.Actions = [
                self.FAMILY_NICHE,
                self.MOVE_FORWARD,
                self.MOVE_LEFT,
                self.MOVE_RIGHT,
                self.MOVE_BACK,
                self.TURN_LEFT,
                self.TURN_RIGHT,
                self.TURN_BACK,
                self.TURN_RANDOM,
                self.FIND_ENEMY,
                self.GO_ENEMY
            ]
            self.Actions.extend([cdef(self.SECRETE_PHEROMONE,ptype) for ptype in ptypes])


        # Use the functions and environment to decide what to do
        def executeActions(self):
            def possible_actions(self) -> list: # Create a list of possible actions a single creature can take depending on the sinkNums in their genome.
                Actions_list = []
                for gene in self.genome.genes:
                    Actions_list.append(self.Actions[gene.sinkNum])
                return Actions_list
            a = possible_actions(self)

            # Get the input values from the sensors
            input_data = [[func() for func in self.Sensors]*len(self.genome.genes)]

            # Pass the input values through the neural network to get the output values
            outputs = self.neural_network.predict_feedforward(input_data).argmax(axis=1)

            # Calculate the probabilities using the softmax function
            probabilities = np.exp(outputs) / np.sum(np.exp(outputs))

            # Choose an action based on the highest probabilities
            action_index = np.random.choice(range(len(a)), p=probabilities)

            # Perform the selected action
            a[action_index]()
            self.hunger += 1
            self.age += 1
            
            self.KILL_FORWARD()
            
            self.REPRODUCE()



    def findSpecies(self):
        for i, species in enumerate(species_list):
            if species == self:
                return i, species


    def __init__(self):
        self.creature_list = []
        self.values = []


    def create_population(self, size):
        self.creature_list = [
            self.Creature(
                random.randint(0, p.grid_width), 
                random.randint(0, p.grid_height), 
                p.speed, 
                p.vision, 
                best_creatures[self.findSpecies()[0]].genome.mutate() if best_creatures != 0 else makeRandomGenome(), 
                random.randrange(4),
                self.findSpecies()[0] - 1,
                self.findSpecies()[1]
                ) 
            for i in alpha(size)
        ]

class Pheromone_type:
    class Pheromone:
        def __init__(self, x, y, ptype):
            self.x = x
            self.y = y
            self.ptype = ptype
            
        def DISSIPATE(self):
            self.x = (self.x + random.uniform(-0.1, 0.1))%p.grid_width
            self.y = (self.y + random.uniform(-0.1, 0.1))%p.grid_height
            if random.random() < 0.1:
                pass
    def __init__(self):
        self.pheromone_list = []
        
def fitness_function():
    foodHighscores = []
    bestCreatures = []
    for i,species in enumerate(species_list):
        bestCreatures.append(0)
        foodHighscores.append(-1)
        for creature in species_list[i].creature_list:
            if creature.foodHighscore > foodHighscores[i]: bestCreatures[i], foodHighscores[i] = creature, creature.foodHighscore
    return bestCreatures
def difference(a, b):
    return a - b
def theta(degrees):
    return degrees * (np.pi/180)
def distance(a, b):
    return np.sqrt(a**2 + b**2) + 0.0000001
def go(lists):
    # Move and eat for creatures
    j = 0
    # Iterates a while loop for every creature in the simulation
    while j < len(lists):
        grid[lists[j].rx][lists[j].ry] = ' '
        lists[j].executeActions()
        # Repeats the loop if a creature dies so the index (j) doesn't go out of range
        if lists[j].hunger >= 30 or lists[j].age > p.maxAge+random.randint(0,20) or lists[j].health < 0:
            lists.pop(j)
        else:
            #put the prey on the grid
            if lists[j].age > (p.maxAge / 4.44444444444):
                grid[lists[j].rx][lists[j].ry] = lists[j].display
                
            else:
                grid[lists[j].rx][lists[j].ry] = str.lower(lists[j].display)
            j += 1


def popu(species):
    return sum(list(map(lambda x: len(x.creature_list), species)))


# Functions for every 'frame' in the simulation
def simulate():
    
    # Activate each species in the species list and clear the previous grid
    for species in species_list:
        go(species.creature_list)
    cls() # Clears screen

    # Create the population variable and print the total populations of each species
    population = 0
    for i,species in enumerate(species_list):
        print(f"{selected[i]}: {len(species.creature_list)}")

    # Print outputs for better understanding]
    print(f"{rst}Population: {popu(species_list)}")
    print(f"Current gen: {gen}")
    print(f"Current sim: {simulationLength}")
    print(f"Longest sim: {simulationLengthHighest}")
    
    # Use mirrored mode if enabled
    if p.mirror == True:
        for row in grid:
            print(*row, sep =' ', end=" ", flush=True)
            print(*row, sep =' ', end=" ", flush=True)
            print()
        return

    # Print grid values
    for row in reversed(grid): print(*row, sep=" ", flush=True)


# Define game variables
simulationLengthHighest = 0
best_creatures = 0
selected = generate_display()
gen = 0

# Main game loop
while True:

    # Define generational variables
    grid_sp = 1
    grid = [[' ' for _ in alpha(p.grid_height+2)] for _ in alpha(p.grid_width+2)]
    simulationLength = 0
    species_list = []
    ptypes = []
    length = []
    gen += 1
    
    for i in alpha(p.pheromones):
        ptypes.append(Pheromone_type())
    
    # Create a species and population for each 
    for i in range(p.biodiversityCount):
        species_list.append(Life())
        species_list[i].create_population(round(p.grid_size/(p.biodiversityCount / p.creatureProportion)))
    
    # Simulation loop for every frame
    breakloop = False
    while True:
        
        # Detect when to reset and go to the next generation
        for species in species_list:
            if len(species.creature_list) < p.minCreatures:
                best_creatures = fitness_function()
                print("Fail!")
                breakloop = True
        if breakloop: break

        # Simulate a frame and increase the simulationLength/time for the pyplot
        simulate()
        simulationLength += 1
        length.append(simulationLength)
        for species in species_list:
            species.values.append(len(species.creature_list))
            plt.plot(length,species.values)
        plt.draw()
        plt.pause(0.01)
        #plt.clf()
        if simulationLength > simulationLengthHighest: simulationLengthHighest = simulationLength
        time.sleep(0.02) # Allows the user to see the simulation images 1 by 1