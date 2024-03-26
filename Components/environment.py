"""
2D Space for individuals within the population to interact with
various environmental features.

Step 1) Setup environment and 1st feature (holes)
"""

# IMPORTS
import numpy as np


# OBJECTS
class Environment:

    def __init__(self, length_units, width_units):

        self.length, self.width = length_units, width_units
        # Feature constants
        self.holes = 20  # Number of holes throughout entire environment

    def build(self):

        # Store space in numpy 2D array
        arr = np.arange(self.length * self.width)
        env_struct = arr.reshape(self.length, self.width)

        # Dictionary to store information about each individual square within environment
        env_info_d = {key: None for key in list(arr)}

        """ ADDING FEATURES
        """
        # Borders
        pass

        # Holes
        pass

# RUN
if __name__ == "__main__":
    obj = Environment(length_units=100, width_units=100)
    obj.build()
