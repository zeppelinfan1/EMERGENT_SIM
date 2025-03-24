# IMPORTS
import numpy as np
import random
from dataclasses import dataclass, field
from collections import OrderedDict
import Components.network as nn
import Components.mapping as mapping

@dataclass
class Genetics:

    gene_number: int
    gene_length: int
    genes: list = field(init=False)
    mapping: list = field(init=False)

    def __post_init__(self):

        # Binary array for each neuron
        self.genes = [self.generate_gene() for _ in range(self.gene_number)]
        # Mapping for each gene
        self.mapping = self.generate_map()

    def generate_gene(self):

        # Random binary list
        return [random.randint(0, 1) for _ in range(self.gene_length)]

    def generate_map(self):

        mapping_list = []
        for gene in self.genes:

            half = len(gene) // 2  # Split gene into two parts
            x_binary = gene[:half]  # First half for X
            y_binary = gene[half:]  # Second half for Y
            # Mapping a binary gene to a Hilbert curve coordinate
            x_decimal = int("".join(map(str, x_binary)), 2)
            y_decimal = int("".join(map(str, y_binary)), 2)
            max_value = (2 ** half) - 1  # Max value for scaling
            x_value = (x_decimal / max_value) * 2 - 1
            y_value = (y_decimal / max_value) * 2 - 1

            mapping_list.append((x_value, y_value))

        return mapping_list


    def __repr__(self):

        # Formats output
        return f"Genetics(gene_number={self.gene_number}, gene_length={self.gene_length}, genes={self.genes}, mapping={self.mapping})"

@dataclass
class Memory:

    max_observations: int = 200
    memory: OrderedDict = field(default_factory=OrderedDict)

    def add(self, embedding, label):

        pass

@dataclass
class Subject:

    id: int = field(init=False)
    # Genetics information
    gene_number: int
    gene_length: int
    # Environmental related paramaters
    perception_range: int
    env_memory: dict = field(default_factory=dict)
    # Subject paramaters
    energy_change: int = 0
    energy: int = 100
    genetics: Genetics = field(init=False) # Created in post init
    # Features network paramaters
    feature_embedding_length: int = 4 # Length of 3 for embeddings + 1 for numerous_features 1 hot encoding = 4
    feature_embeddings: dict = field(default_factory=dict)
    feature_network: nn.Model = field(init=False)
    feature_mapping: dict = field(default_factory=dict)
    feature_memory: Memory = field(init=False)

    last_subject = 0

    def __post_init__(self):

        Subject.last_subject += 1
        self.id = Subject.last_subject
        self.genetics = Genetics(gene_number=self.gene_number, gene_length=self.gene_length)
        # Network initialization
        self.feature_network = self.initialize_network()

    def generate_new_embedding(self, name, length=3):

        new_embedding = np.random.uniform(low=0.0, high=1.0, size=(length,))
        # Add to dict
        self.feature_embeddings[name] = new_embedding

    def initialize_network(self):

        # Instantiate the model
        network = nn.Model()

        # Add layers
        network.add(nn.Layer_Dense(self.feature_embedding_length, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
        network.add(nn.Activation_ReLU())
        network.add(nn.Layer_Dense(512, 512))
        network.add(nn.Activation_ReLU())
        network.add(nn.Layer_Dropout(rate=0.1))
        network.add(nn.Layer_Dense(512, 4))

        # Set loss, optimizer and accuracy objects
        network.set(
            loss=nn.Loss_Constrastive(),
            optimizer=nn.Optimizer_Adam(decay=1e-7),
            accuracy=nn.Accuracy_Constrastive()
        )

        network.finalize()

        return network

    def update_memory(self, env_section):

        # Loop through each square and replace with current
        self.env_memory.update(env_section)

if __name__ == "__main__":

    subject1 = Subject(gene_number=6, gene_length=10, perception_range=2)
    subject1.generate_new_embedding(name="TEST", length=3)
    print(subject1.feature_embeddings)

