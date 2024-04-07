"""
INDIVIDUAL SUBJECTS
"""

# IMPORTS
from Components.nn_brain import Brain


# OBJECTS
class Subject:

    def __init__(self, number: int):

        # Id number
        self.id_number = number

    # Initializing neural network brain
    nn_brain = Brain()


# RUN
if __name__ == "__main__":
    obj = Subject(number=100)

