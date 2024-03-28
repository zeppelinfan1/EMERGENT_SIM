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

    def borders(self, dict, struct):

        # Loop through each key
        for key in dict.keys():

            # Identify coordinated
            y, x = np.where(struct == key)
            y, x = y[0], x[0]

            # Top border check
            if (y - 1) < 0:
                dict[key].append("TOP_BORDER")
            # Bottom border check
            if (y + 1) > self.length - 1:
                dict[key].append("BOTTOM_BORDER")
            # Left border check
            if (x - 1) < 0:
                dict[key].append("LEFT_BORDER")
            # Right border check
            if (x + 1) > self.width - 1:
                dict[key].append("RIGHT_BORDER")

            print(dict[key])

        return dict

    def build(self):

        # Store space in numpy 2D array
        arr = np.arange(self.length * self.width)
        env_struct = arr.reshape(self.length, self.width)

        # Dictionary to store information about each individual square within environment
        env_info_d = {key: [] for key in list(arr)}

        """ ADDING FEATURES
        """
        # Borders
        env_info_d = self.borders(env_info_d, env_struct)

        # Holes
        pass

# RUN
if __name__ == "__main__":
    obj = Environment(length_units=100, width_units=100)
    obj.build()
