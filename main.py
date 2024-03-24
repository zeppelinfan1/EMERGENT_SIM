"""
ECOSYSTEM EVOLUTION SIMULATOR
"""

# INITIAL CONSTANTS
initial_pop = 200


# IMPORTS
import pandas as pd, numpy as np
from Components.genetics import GENE


# OBJECTS
class SIM:

    def __init__(self):

        """CONSTANTS
        """
        # Population
        self.pop_high = 200
        self.pop_low = 100
        # Individuals
        self.num_genes = 10
        # Environment
        self.height, self.width = 100, 100

        # Components
        self.gene = GENE()

    def build_pop(self, df: pd.DataFrame, number: int) -> None:

        # Build individuals
        for ind in range(number):

            arr = self.build_ind()

            # Append to main df
            df = pd.concat(objs=[df, arr], ignore_index=True)

        print(df)

    def build_ind(self) -> pd.array:

        # Random genetics
        gene_list = []
        for i in range(self.num_genes):

            gene_list.append(self.gene.build())

        # Random start coords
        x = np.random.randint(0, self.width + 1)
        y = np.random.randint(0, self.height + 1)

        # Generate return array
        arr = pd.DataFrame(data={"genetics": [",".join(gene_list)], "x_cord": [x], "y_cord": [y]})

        return arr

    def start(self) -> None:

        # Build initial population
        self.build_pop(df=pd.DataFrame(), number=initial_pop)


# MAIN OBJECT
def main():

    # Main instance of simulation object
    sim = SIM()
    sim.build_pop(df=pd.DataFrame(columns=["genetics", "x_cord", "y_cord"]), number=initial_pop)


# RUN
if __name__ == "__main__":
    main()
