"""
ECOSYSTEM EVOLUTION SIMULATOR
"""

# INITIAL CONSTANTS
initial_pop = 200
env_length = 100
env_width = 100

# IMPORTS
from Components.environment import Environment
from Components.subject import Subject


# OBJECTS
def build_pop(number: int) -> dict:

    # Initial population dictionary
    pop_dict = {}
    # Build individuals
    for id_num in range(number):

        # Subject object
        subject = Subject(id_num)
        # Append to dictionary
        pop_dict[id_num] = subject

    return pop_dict


class SIM:

    env = Environment(length_units=env_length, width_units=env_width).build()
    # Subjects
    pop = build_pop(number=initial_pop)


# MAIN OBJECT
def main():

    # Main instance of simulation object
    sim = SIM()

    # Main loop
    while True:

        # Temporary
        return


# RUN
if __name__ == "__main__":
    main()
