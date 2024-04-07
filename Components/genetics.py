"""
GENETIC MATERIAL
Provides possible actions for neural network brain to process.
Actions are associated with various properties within the ecosystem.
"""

# IMPORTS
import numpy as np


# OBJECTS
class Gene:

    def __init__(self):

        # Constants
        self.length = 10

    def build(self) -> str:

        # Random binary string
        gene = "".join(np.random.randint(low=0, high=2, size=self.length).astype(str))

        return gene


# RUN
if __name__ == "__main__":
    gene = Gene()
    gene.build()
