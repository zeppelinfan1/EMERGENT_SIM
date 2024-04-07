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
        self.num_holes = 20  # Number of holes throughout entire environment

    def borders(self, env_dict, struct):

        # Loop through each key
        for key in env_dict.keys():

            # Identify coordinated
            y, x = np.where(struct == key)
            y, x = y[0], x[0]

            # Top border check
            if (y - 1) < 0: env_dict[key].append("BT")
            # Bottom border check
            if (y + 1) > self.length - 1: env_dict[key].append("BB")
            # Left border check
            if (x - 1) < 0: env_dict[key].append("BL")
            # Right border check
            if (x + 1) > self.width - 1: env_dict[key].append("BR")

        return env_dict

    def holes(self, env_dict):

        holes = 0
        while holes < self.num_holes:

            # Pick random site
            site_num = np.random.randint(low=0, high=len(env_dict) - 1)
            # Ensure hole feature hasn't already been added
            if "H" not in env_dict[site_num]:
                # Add hole keyword "H"
                env_dict[site_num].append("H")
                holes += 1

        return env_dict

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
        env_info_d = self.holes(env_info_d)

        return env_info_d

# RUN
if __name__ == "__main__":
    obj = Environment(length_units=100, width_units=100)
    obj.build()
