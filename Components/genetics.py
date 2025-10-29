"""
GENETIC MATERIAL
Provides possible actions for neural network brain to process.
Actions are associated with various properties within the ecosystem.
"""

# IMPORTS
import numpy as np
import random
from dataclasses import dataclass, field


# OBJECTS
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

    @staticmethod
    def bits_to_unit(b):

        n = len(b)
        if n == 0:
            return 0.0
        val = int("".join(map(str, b)), 2)
        max_val = (1 << n) - 1

        return (val / max_val) * 2.0 - 1.0 if max_val > 0 else 0.0

    def generate_gene(self):

        # Random binary list
        return [random.randint(0, 1) for _ in range(self.gene_length)]

    def generate_map(self):

        mapping_list = []
        for gene in self.genes:

            n = len(gene)
            seg = n // 3 or 1
            x_bits = gene[:seg]
            y_bits = gene[seg:2*seg]
            z_bits = gene[2*seg:]
            # Mapping a binary gene to a Hilbert curve coordinate
            x_value = self.bits_to_unit(x_bits)
            y_value = self.bits_to_unit(y_bits)
            z_value = self.bits_to_unit(z_bits)
            mapping_list.append((x_value, y_value, z_value))

        return mapping_list


    def __repr__(self):

        # Formats output
        return f"Genetics(gene_number={self.gene_number}, gene_length={self.gene_length}, genes={self.genes}, mapping={self.mapping})"


# RUN
if __name__ == "__main__":
    gene = Genetics(gene_number=3, gene_length=6)
    print(gene.genes)
    print(gene.mapping)
