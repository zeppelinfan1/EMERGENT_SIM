# IMPORTS
import numpy as np
import random
from dataclasses import dataclass, field
import Components.nn_brain as nn

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
class Subject:

    id: int = field(init=False)
    gene_number: int
    gene_length: int
    features: int
    energy: int = 100
    brain: nn.Model = field(init=False)
    genetics: Genetics = field(init=False) # Created in post init

    last_subject = 0

    def __post_init__(self):

        Subject.last_subject += 1
        self.id = Subject.last_subject
        self.genetics = Genetics(gene_number=self.gene_number, gene_length=self.gene_length)
        self.brain = self.initialize_brain()

    def initialize_brain(self):

        # Instantiate the model
        brain = nn.Model()

        # Add layers
        brain.add(nn.Layer_Dense(self.features, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
        brain.add(nn.Activation_ReLU())
        brain.add(nn.Layer_Dense(512, 512))
        brain.add(nn.Activation_ReLU())
        brain.add(nn.Layer_Dropout(rate=0.1))
        brain.add(nn.Layer_Dense(512, 9))
        brain.add(nn.Activation_Softmax())

        # Set loss, optimizer and accuracy objects
        brain.set(
            loss=nn.Loss_CategoricalCrossentropy(),
            optimizer=nn.Optimizer_Adam(decay=1e-7),
            accuracy=nn.Accuracy_Categorical()
        )

        # Finalize
        brain.finalize()

        return brain


if __name__ == "__main__":
    subject1 = Subject(gene_number=6, gene_length=10, features=4)
    print(subject1)


