# Life Simulation/Neuro-Grid 0.0.1
This is a simulation of a virtual ecosystem that allows the user to observe how different species interact with each other. The simulation is written in Python and uses various classes and functions to represent creatures, genomes, and neural networks.

## Table of Contents
- [Installation](https://github.com/GlassOfUnstableMilk/Neuro-Grid/tree/main#installation)
- [Usage](https://github.com/GlassOfUnstableMilk/Neuro-Grid/tree/main#usage)
- [Classes](https://github.com/GlassOfUnstableMilk/Neuro-Grid/tree/main#classes)
- [Class: Gene](https://github.com/GlassOfUnstableMilk/Neuro-Grid/tree/main#gene)
- [Class: Genome](https://github.com/GlassOfUnstableMilk/Neuro-Grid/tree/main#genome)
- [Class: NeuralNetwork](https://github.com/GlassOfUnstableMilk/Neuro-Grid/tree/main#neuralnetwork)
- [Class: Life](https://github.com/GlassOfUnstableMilk/Neuro-Grid/tree/main#life)
- [Requirements](https://github.com/GlassOfUnstableMilk/Neuro-Grid/tree/main#requirements)

## Installation
Use your clipboard to crtl+c and crtl+v the Python file onto your prefered IDE.

## Usage

```python
class Params:
    def __init__(self):
        # This acts as a configuration object that holds various settings and parameters for the simulation
        self.minGenomeLength = 2 # Minimum intelligence level for the neural network
        self.maxGenomeLength = 32 # Maximum intelligence level for the neural network
        self.maxAge = 80 # Doesn't do anything
        self.grid_height = 10 # width
        self.grid_width = 10 # height
        self.grid_size = round((self.grid_height * self.grid_width) / ((self.grid_height + self.grid_width) / 2)) # Don't change
        self.speed = 1 # How much creatures move each time 
        self.biodiversityCount = 3 # How many species total
        self.creatureProportion = 3 # How many creatures are created depending on the grid_size
        self.minCreatures = 5 # The minimum value of any population until the simulation resets to the next generation
        self.mutationRate = 2 # How much a creature can mutate while reproducing
        self.mirror = False # Due to the borders of the grid using modulo, mirror prints the grid twice, seamlessly
```


## Classes

### Gene
The Gene class represents a single gene and holds information about a source and sink number.
```python
class Gene:
    def __init__(self,): # Sink number represents the actions a gene could take
        self.sourceNum = random.randint(0,7)
        self.sinkNum = random.randint(0,12)
    def makeCustomGene(self, source, sink, weight):
        self.sourceNum = source
        self.sinkNum = sink
        self.weight = weight
        
def makeRandomGene():
    return Gene()
```

### Genome
The Genome class is a collection of genes and is used to create a neural network that predicts an output for a given input. Basically, the Gene class but bigger.
```python
class Genome:
    def __init__(self):
        self.genes = []

    def add_gene(self, gene):
        self.genes.append(gene)
        
    def mutate(self):
        for i in range(random.randrange(p.mutationRate)): # It loops a random number of times based on mutationRate
            if random.random() <= 0.5:
                mutated_gene = random.randrange(len(self.genes))
                self.genes[mutated_gene].weight = random.uniform(-4.0, 4.0) # It randomly selects a gene in the genes list sets the weight to a random value
        return self
        

def makeRandomGenome(): # Creates a genome, and appends the Gene class to the genome: 'genes' list.
    genome = Genome()
    length = random.randint(p.minGenomeLength, p.maxGenomeLength)
    for _ in range(length):
        genome.add_gene(makeRandomGene())
    return genome
```

### NeuralNetwork
The NeuralNetwork class takes a genome object as input and creates a neural network based on the genes in the genome. The network performs matrix calculations using the numpy library to predict the output.
```python
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
```

### Life
The Life class is used to simulate the behavior and actions of creatures in a simulation. It contains an inner class called "Creature" that has methods to detect other creatures, move randomly or in a specific direction, reproduce, and kill a creature in front of it. The methods also keep track of a creature's hunger, age, and health, and the reproduction function depends on these factors and a random chance.
```python
class Life:

    class Creature:
            
        # Find the distance from an alien
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
            
        # Find the distance from family
        def DETECTION_FAMILY_RADIUS(self):
            distances = []
            for i in range(len(species_list[self.index].creature_list)):
                x = (p.grid_width - abs(self.x - species_list[self.index].creature_list[i].x)) / p.grid_width
                y = (p.grid_height - abs(self.y - species_list[self.index].creature_list[i].y)) / p.grid_height
                distances.append((x + y)/2)
            distances.sort(reverse=True)
            return distances[0] if len(species_list[self.index].creature_list) > 0 else 0
            
        # Change coordinates randomly
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
            
        # Changes a single coordinate to the modulo of the grid
        def MOVE_EAST(self):
            self.x = (self.x + self.speed)%p.grid_width
            self.hunger += 1
            self.age += 1
            
        
        def MOVE_WEST(self):
            self.x = (self.x - self.speed)%p.grid_width
            self.hunger += 1
            self.age += 1
            
        # Increases y to the mod of grid
        def MOVE_NORTH(self):
            self.y = (self.y + self.speed)%p.grid_height
            self.hunger += 1
            self.age += 1
            
        
        def MOVE_SOUTH(self):    
            self.y = (self.y - self.speed)%p.grid_height
            self.hunger += 1
            self.age += 1

        # Duplicates and mutates an offspring if the following requirements are met
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

        # This is how the creatures eat and get food
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
            
        #  Randomly select an input from 1 to 0
        def RANDOM(self):
            return random.random()

        # Make an input depending on how extinct their food is
        def ECO_NICHE(self):
            try:
                return (p.grid_size/5) / len(species_list[self.enemy_index].creature_list) 
            except ZeroDivisionError: 
                return 0
        
        # If their last moves are the same as their direction
        def LAST_MOVE_NORTH(self):
            return 1 if self.orientation[self.last_dir] == "NORTH" else 0
            
        def LAST_MOVE_EAST(self):
            return 1 if self.orientation[self.last_dir] == "EAST" else 0
            
        def LAST_MOVE_SOUTH(self):
            return 1 if self.orientation[self.last_dir] == "SOUTH" else 0
            
        def LAST_MOVE_WEST(self):
            return 1 if self.orientation[self.last_dir] == "WEST" else 0
        
        # Rotate directions
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

        # Move depending on the direction
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

        Use the functions and environment to decide what to do
        def executeActions(self):
            def possible_actions(self) -> list: # Create a list of possible actions a single creature can take depending on the sinkNums in their genome.
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
```

## Requirements
- Python - the script is written in Python, a high-level, interpreted programming language.

To install go to [python.org](https://www.python.org/downloads/) and find your current release.

- [Object-Oriented Programming (OOP)](https://en.wikipedia.org/wiki/Object-oriented_programming) - a programming paradigm that uses objects to model real-world entities and their behaviors. The code is structured using OOP, creating several classes such as Params, Gene, Genome, NeuralNetwork, Life, and Creature, to simulate the behavior and actions of creatures in a virtual ecosystem.

Most of everything in Python is an object. From functions to classes, this knowledge is required

- [Numpy](https://numpy.org) - a library for the Python programming language, used for working with arrays and matrices of numerical data. The code uses Numpy to perform matrix calculations for the neural network simulation.

This should be automatically downloaded when you install Python.

- [Neural Networking](https://en.wikipedia.org/wiki/Neural_network) - a type of machine learning algorithm modeled after the structure and function of the human brain, used to solve complex problems. The code creates a neural network object using the Genome class, which takes a genome object as input and creates a neural network based on the genes in that genome. The network predicts an output for a given input data.

Some knowledge on algebra, and how a brain functions is necessary.

- [ANSI Escape Codes](https://en.wikipedia.org/wiki/ANSI_escape_code) - a protocol for controlling the formatting, color, and other output options on a computer terminal. The code uses ANSI escape codes to generate a random color-character combination for display purposes.

Optional, if you want to edit the possible colors and formatting.
