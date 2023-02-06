import random
import time
import numpy as np
from console.utils import cls

def imprint():
    pass

class Params:
    def __init__(self):
        self.minGenomeLength = 2
        self.maxGenomeLength = 32
        self.maxAge = 80
        self.grid_height = 150 # width
        self.grid_width = 50 # height
        self.grid_size = round((self.grid_height * self.grid_width) / ((self.grid_height + self.grid_width) / 2))
        self.speed = 1
        self.biodiversityCount = 5
        self.creatureProportion = 3
        self.minCreatures = 5
        self.mutationRate = 2
        self.mirror = False

p = Params()

import random

def generate_display():
    # list of uppercase alphabet characters
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    # list of ANSI escape sequence colors (roy g biv)
    colors = ['\033[31m', '\033[32m', '\033[33m', '\033[34m', '\033[35m', '\033[36m']

    # list to store selected characters and colors
    fselected = []

    # loop to select randok characters and colors
    for i in range(p.biodiversityCount):
        selected_char = random.choice(alphabet)
        alphabet.remove(selected_char)
        selected_color = random.choice(colors)
        colors.remove(selected_color)
        fselected.append(selected_color + selected_char)
    return fselected

class Gene:
    def __init__(self,):
        self.sourceNum = random.randint(0,7)
        self.sinkNum = random.randint(0,12)
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
        for i in range(random.randrange(p.mutationRate)):
            if random.random() <= 0.5:
                mutated_gene = random.randrange(len(self.genes))
                self.genes[mutated_gene].weight = random.uniform(-4.0, 4.0)
        return self
        

def makeRandomGenome():
    genome = Genome()
    length = random.randint(p.minGenomeLength, p.maxGenomeLength)
    for _ in range(length):
        genome.add_gene(makeRandomGene())
    return genome

class NeuralNetwork:
    
    def create_network(self):
        # initialize weights and biases
        for i in range(len(self.genome.genes)):
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


class Life:

    class Creature:
            

        def DETECTION_ALIEN_RADIUS(self):
            distances = []
            for j in range(len(species_list)):
                if j == self.index:
                    continue
                for i in range(len(species_list[j].creature_list)):
                    x = (p.grid_width - abs(self.x - species_list[j].creature_list[i].x)) / p.grid_width
                    y = (p.grid_height - abs(self.y - species_list[j].creature_list[i].y)) / p.grid_height
                    distances.append((x + y)/2)
            distances.sort(reverse=True)
            return distances[0] if len([i for i in range(len(species_list)) if i != self.index]) > 0 else 0
            

        def DETECTION_FAMILY_RADIUS(self):
            distances = []
            for i in range(len(species_list[self.index].creature_list)):
                x = (p.grid_width - abs(self.x - species_list[self.index].creature_list[i].x)) / p.grid_width
                y = (p.grid_height - abs(self.y - species_list[self.index].creature_list[i].y)) / p.grid_height
                distances.append((x + y)/2)
            distances.sort(reverse=True)
            return distances[0] if len(species_list[self.index].creature_list) > 0 else 0
            

        def MOVE_RANDOM(self):
            self.x = (self.x + random.randint(-self.speed, self.speed))%p.grid_width
            self.y = (self.y + random.randint(-self.speed, self.speed))%p.grid_height
            self.hunger += 1
            self.age += 1 
            
        def MOVE_X(self):
            self.x = (self.x + random.randint(-self.speed, self.speed))%p.grid_width
            self.hunger += 1
            self.age += 1 
            
        def MOVE_Y(self):
            self.y = (self.y + random.randint(-self.speed, self.speed))%p.grid_height
            self.hunger += 1
            self.age += 1 
            

        def MOVE_EAST(self):
            self.x = (self.x + self.speed)%p.grid_width
            self.hunger += 1
            self.age += 1
            

        def MOVE_WEST(self):
            self.x = (self.x - self.speed)%p.grid_width
            self.hunger += 1
            self.age += 1
            

        def MOVE_NORTH(self):
            self.y = (self.y + self.speed)%p.grid_height
            self.hunger += 1
            self.age += 1
            

        def MOVE_SOUTH(self):    
            self.y = (self.y - self.speed)%p.grid_height
            self.hunger += 1
            self.age += 1

        def REPRODUCE(self):
            if self.hunger < 10 and self.age > 8 and self.health > 0 and random.random() <= 0.5:
                species_list[self.index].creature_list.append(
                    species_list[self.index].Creature(
                        (self.x + random.randint(-1, 1))%p.grid_width, 
                        (self.y + random.randint(-1, 1))%p.grid_height, 
                        self.speed, self.vision,
                        self.genome.mutate(), 
                        self.direction, 
                        self.enemy_index
                    )
                )


        def KILL_FORWARD(self):
            for i in range(len(species_list[self.enemy_index].creature_list)):
                if abs(self.x - species_list[self.enemy_index].creature_list[i].x) <= self.vision and abs(self.y - species_list[self.enemy_index].creature_list[i].y) <= self.vision:
                    grid[species_list[self.enemy_index].creature_list[i].x][species_list[self.enemy_index].creature_list[i].y] = " "
                    species_list[self.enemy_index].creature_list.pop(i)
                    self.hunger = 0
                    self.foodHighscore += 1
                    self.health += 5
                    if self.health > 100: self.health = 100
                    return True
            return False
            

        def RANDOM(self):
            return random.random()


        def ECO_NICHE(self):
            try:
                return (p.grid_size/5) / len(species_list[self.enemy_index].creature_list) 
            except ZeroDivisionError: 
                return 0
        
                
        def LAST_MOVE_NORTH(self):
            return 1 if self.orientation[self.last_dir] == "NORTH" else 0
            
        def LAST_MOVE_EAST(self):
            return 1 if self.orientation[self.last_dir] == "EAST" else 0
            
        def LAST_MOVE_SOUTH(self):
            return 1 if self.orientation[self.last_dir] == "SOUTH" else 0
            
        def LAST_MOVE_WEST(self):
            return 1 if self.orientation[self.last_dir] == "WEST" else 0
        

        def ROTATE_RIGHT(self):
            if self.direction + 1 > 3:
                self.direction = 0
            else:    
                self.direction += 1


        def ROTATE_LEFT(self):
            if self.direction - 1 < 0:
                self.direction = 3
            else:
                self.direction -= 1


        def MOVE_FORWARD(self):
            if self.orientation[self.direction] == "NORTH":
                self.y = (self.y + self.speed)%p.grid_height

            elif self.orientation[self.direction] == "EAST":
                self.x = (self.x + self.speed)%p.grid_width
            
            elif self.orientation[self.direction] == "WEST":
                self.x = (self.x - self.speed)%p.grid_width

            elif self.orientation[self.direction] == "SOUTH":
                self.y = (self.y - self.speed)%p.grid_height
                
            self.hunger += 1
            self.age += 1
            self.last_dir = self.direction
                    
        def MOVE_REVERSE(self):
            if self.orientation[self.direction] == "NORTH":
                self.y = (self.y - self.speed)%p.grid_height
                self.hunger += 1
                self.age += 1
                self.last_dir = self.direction + 2

            elif self.orientation[self.direction] == "EAST":
                self.x = (self.x - self.speed)%p.grid_width
                self.hunger += 1
                self.age += 1
                self.last_dir = self.direction + 2
            
            elif self.orientation[self.direction] == "WEST":
                self.x = (self.x + self.speed)%p.grid_width
                self.hunger += 1
                self.age += 1
                self.last_dir = self.direction - 2

            elif self.orientation[self.direction] == "SOUTH":
                self.y = (self.y + self.speed)%p.grid_height
                self.hunger += 1
                self.age += 1
                self.last_dir = self.direction - 2
                
        def MOVE_LEFT(self):
            if self.orientation[self.direction] == "NORTH":
                self.x = (self.x - self.speed)%p.grid_width

            elif self.orientation[self.direction] == "EAST":
                self.y = (self.y + self.speed)%p.grid_height
            
            elif self.orientation[self.direction] == "WEST":
                self.y = (self.y - self.speed)%p.grid_height

            elif self.orientation[self.direction] == "SOUTH":
                self.x = (self.x + self.speed)%p.grid_width
            
            self.hunger += 1
            self.age += 1
            self.last_dir = self.direction - 1
            
        def MOVE_RIGHT(self):
            if self.orientation[self.direction] == "NORTH":
                self.x = (self.x + self.speed)%p.grid_width

            elif self.orientation[self.direction] == "EAST":
                self.y = (self.y - self.speed)%p.grid_height
            
            elif self.orientation[self.direction] == "WEST":
                self.y = (self.y + self.speed)%p.grid_height

            elif self.orientation[self.direction] == "SOUTH":
                self.x = (self.x - self.speed)%p.grid_width
            
            self.hunger += 1
            self.age += 1
            self.last_dir = self.direction + 1
                
        def MOVE_LEFTandRIGHT(self):
            
            direction = random.randint(-self.speed, self.speed)
            
            if self.orientation[self.direction] == "NORTH" or self.orientation[self.direction] == "SOUTH":
                self.x = (self.x + direction)%p.grid_width
                self.hunger += 1
                self.age += 1
                self.last_dir = self.direction - 1 if direction < 0 else self.direction + 1
                
            elif self.orientation[self.direction] == "EAST" or self.orientation[self.direction] == "WEST":
                self.y = (self.y + direction)%p.grid_height
                self.hunger += 1
                self.age += 1
                self.last_dir = self.direction - 1 if direction < 0 else self.direction + 1 
            
        
        def __init__(self, x: int, y: int, speed: int, vision: int, genome: Genome, direction: int, enemy_index: int):
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
            self.orientation = ["NORTH", "EAST", "SOUTH", "WEST"]
            self.enemy_index = enemy_index
            self.index = (self.enemy_index) - 1
            self.display = selected[self.index]
            self.foodHighscore = 0
            
            # Initialize the neural network with the weights and biases from the genome
            self.neural_network = NeuralNetwork(self.genome)

            self.Sensors = [
                self.DETECTION_ALIEN_RADIUS,
                self.DETECTION_FAMILY_RADIUS,
                self.RANDOM,
                self.ECO_NICHE,
                self.LAST_MOVE_NORTH,
                self.LAST_MOVE_EAST,
                self.LAST_MOVE_SOUTH,
                self.LAST_MOVE_WEST,
            ]
            
            self.Actions = [
                self.MOVE_RANDOM,
                self.MOVE_EAST,
                self.MOVE_WEST,
                self.MOVE_NORTH,
                self.MOVE_SOUTH,
                self.KILL_FORWARD,
                self.MOVE_FORWARD,
                self.ROTATE_LEFT,
                self.ROTATE_RIGHT,
                self.MOVE_REVERSE,
                self.MOVE_LEFT,
                self.MOVE_RIGHT,
                self.MOVE_LEFTandRIGHT,
            ]            


        def executeActions(self):
            def possible_actions(self) -> list:
                Actions_list = []
                for gene in self.genome.genes:
                    Actions_list.append(self.Actions[gene.sinkNum])
                return Actions_list
            a = possible_actions(self)

            # Get the input values from the sensors
            input_data = [[self.DETECTION_ALIEN_RADIUS(), self.DETECTION_FAMILY_RADIUS(), self.RANDOM(), self.ECO_NICHE()]*len(self.genome.genes)]

            # Pass the input values through the neural network to get the output values
            outputs = self.neural_network.predict_feedforward(input_data).argmax(axis=1)

            # Calculate the probabilities using the softmax function
            probabilities = np.exp(outputs) / np.sum(np.exp(outputs))

            # Choose an action based on the highest probabilities
            action_index = np.random.choice(range(len(a)), p=probabilities)

            # Perform the selected action
            a[action_index]()
            self.REPRODUCE()

    def findIndex(self):
        for i in range(len(species_list)):
            if species_list[i] == self:
                return i

    def __init__(self):
        self.creature_list = []

    def create_population(self, size):
        self.creature_list = [self.Creature(random.randint(0, p.grid_width), random.randint(0, p.grid_height), p.speed, 0, best_creatures[self.findIndex()].genome.mutate() if best_creatures != 0 else makeRandomGenome(), random.randrange(4), self.findIndex() - 1) for i in range(size)]


def fitness_function():
    foodHighscores = []
    bestCreatures = []
    for i in range(len(species_list)):
        bestCreatures.append(0)
        foodHighscores.append(-1)
        for creature in species_list[i].creature_list:
            if creature.foodHighscore > foodHighscores[i]:
                bestCreatures[i] = creature
                foodHighscores[i] = creature.foodHighscore
    return bestCreatures


def go(lists):
    # Move and eat for creatures
    j = 0
    # Iterates a while loop for every creature in the simulation
    while j < len(lists):
        grid[lists[j].x][lists[j].y] = ' '
        lists[j].executeActions()
        # Repeats the loop if a creature dies so the index (j) doesn't go out of range
        if lists[j].hunger >= 30 or lists[j].age > p.maxAge+random.randint(0,20) or lists[j].health < 0:
            lists.pop(j)
        else:
            #put the prey on the grid
            if lists[j].age > (p.maxAge / 4.44444444444):
                grid[lists[j].x][lists[j].y] = lists[j].display
                
            else:
                grid[lists[j].x][lists[j].y] = str.lower(lists[j].display)
            j += 1
            
# Pheromones
# Multicellular functionality
# Generations


def simulate():
    # go(predator, prey)
    for species in species_list:
        go(species.creature_list)
    cls() # Clears screen
    for species in species_list:
        print(len(species.creature_list))
    print(simulationLengthHighest)
    if p.mirror == True:
        for row in grid:
            print(*row, sep =' ', end=" ", flush=True)
            print(*row, sep =' ', end=" ", flush=True)
            print()
        return
    for row in grid:
        print(*row, sep =' ', flush=True)

grid = []
simulationLengthHighest = 0
best_creatures = 0
selected = generate_display()

while True:

    grid_sp = 1
    grid = [[' ' for _ in range(p.grid_height+2)] for _ in range(p.grid_width+2)]
    simulationLength = 0
    species_list = []
    
    for i in range(p.biodiversityCount):
        species_list.append(Life())
        species_list[i].create_population(round(p.grid_size/(p.biodiversityCount / p.creatureProportion)))

    breakloop = False
    while True:
        for species in species_list:
            if len(species.creature_list) < p.minCreatures:
                best_creatures = fitness_function()
                print("Fail!")
                breakloop = True
        if breakloop:
            break
        simulate()
        simulationLength += 1
        if simulationLength > simulationLengthHighest: simulationLengthHighest = simulationLength
        time.sleep(0.03) # Allows the user to see the simulation images 1 by 1
