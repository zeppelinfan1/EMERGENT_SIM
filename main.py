"""
ECOSYSTEM EVOLUTION SIMULATOR
"""

# INITIAL CONSTANTS
initial_pop = 200
env_length = 100
env_width = 100

# IMPORTS
import pandas as pd, numpy as np
from Components.environment import Environment
from Components.subject import Subject


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

    def build_pop(self, number: int) -> dict:

        # Initial population dictionary
        pop_dict = {}
        # Build individuals
        for id_num in range(number):

            # Subject object
            subject = Subject(id_num)
            # Append to dictionary
            pop_dict[id_num] = subject

        return pop_dict

    def start(self) -> None:

        # Build environment
        env_dict = Environment(env_length, env_width).build()

        # Build initial population
        pop_dict = self.build_pop(number=initial_pop)

        for key_num in pop_dict.keys():

            obj = pop_dict[key_num]
            print(obj.id_number)

# MAIN OBJECT
def main():

    # Main instance of simulation object
    sim = SIM()
    sim.start()


# RUN
if __name__ == "__main__":
    main()
