# IMPORTS
import numpy as np
import random
from dataclasses import dataclass, field


# DATACLASSES
@dataclass
class Genetics:

    gene_number: int
    gene_length: int
    genes: list = field(init=False)

    def __post_init__(self):

        # Binary array for each neuron
        self.genes = [self.generate_gene() for _ in range(self.gene_number)]

    def generate_gene(self):

        # Random binary list
        return [random.randint(0, 1) for _ in range(self.gene_length)]

    def __repr__(self):

        # Formats output
        return f"Genetics(gene_number={self.gene_number}, gene_length={self.gene_length}, genes={self.genes})"

@dataclass
class Subject:

    id: int = field(init=False)
    gene_number: int
    gene_length: int
    genetics: Genetics = field(init=False) # Created in post init

    last_subject = 0

    def __post_init__(self):

        Subject.last_subject += 1
        self.id = Subject.last_subject
        self.genetics = Genetics(gene_number=self.gene_number, gene_length=self.gene_length)


subject1 = Subject(gene_number=6, gene_length=10)
print(subject1)




