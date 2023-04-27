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
#...
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
#...
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
#...
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
#...
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
