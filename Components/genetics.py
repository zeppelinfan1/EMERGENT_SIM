"""
GENETIC MATERIAL
Provides possible actions for neural network brain to process.
Actions are associated with various properties within the ecosystem.
"""

# IMPORTS
import numpy as np
import random, math
from dataclasses import dataclass, field



# OBJECTS
@dataclass(frozen=True)
class Genetics:

    gene_number: int
    gene_length: int
    genes: list = field(init=False)
    genetic_projection: dict = field(init=False)

    def __post_init__(self):

        # Each gene is a binary array which will determine action's performance
        genes = [self.generate_genes() for _ in range(self.gene_number)]
        object.__setattr__(self, "genes", genes)
        # How genetics are interpretted by the Actions plane - genetic Projection
        genetic_projection = self.generate_projections()
        object.__setattr__(self, "genetic_projection", genetic_projection)

    @staticmethod
    def bits_to_value(bits):

        if not bits:
            return 0.0
        # interpret as signed integer centered at 0
        val = int("".join(map(str, bits)), 2)
        mid = (1 << len(bits)) / 2

        return val - mid

    @staticmethod
    def gene_density(gx, gy, strength, x, y, spread: float=1):

        # Contribution of a single gene at point (x, y)
        dist_sq = (x - gx) ** 2 + (y - gy) ** 2

        return abs(strength) * math.exp(-dist_sq / (2 * spread ** 2))

    def generate_genes(self):

        # Random binary list
        return [random.randint(0, 1) for _ in range(self.gene_length)]

    def generate_projections(self):

        projection_dict = {}
        for i, gene in enumerate(self.genes):

            n = len(gene)
            seg = n // 3 or 1

            # Split gene into segments
            x_bits = gene[:seg]
            y_bits = gene[seg:2*seg]
            z_bits = gene[2*seg:]

            # Putting each segment through the alogorithm
            projection_dict[i] = {
                "x": self.bits_to_value(x_bits), # X-coordinate
                "y": self.bits_to_value(y_bits), # y-coordinate
                "z": self.bits_to_value(z_bits), # Magnitude
            }

        return projection_dict

    def density_at(self, x, y):

        # Total genetic density at action coordinate (x, y)
        total = 0.0
        for g in self.genetic_projection.values():

            total += self.gene_density(g["x"], g["y"], g["z"], x, y)

        return total

# RUN
if __name__ == "__main__":
    g = Genetics(gene_number=8, gene_length=10)
    print(g.genetic_projection)
    print(g.density_at(x=1, y=0))
    # Step 3) How action Performance (i.e. with variablility included) differs from the action Potential
