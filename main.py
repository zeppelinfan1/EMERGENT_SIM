"""
ECOSYSTEM EVOLUTION SIMULATOR
"""

MAX_ITERATIONS = 1
ENV_HEIGHT = 50
ENV_WIDTH = 100
SUBJECT_NUM = 10
SUBJECT_GENE_NUM = 8
SUBJECT_GENE_LEN = 10
SUBJECT_PERCEPTION_RANGE = 3


# IMPORTS
import pandas as pd, numpy as np
from Components.environment import Environment
from Components.subject import Subject
from Components.network import Model
from Components.db_api import DB_API


# MAIN OBJECT
def main():

    """INITIALIZE ENVIRONMENT AND SUBJECTS
    """
    env = Environment(width=ENV_WIDTH, height=ENV_HEIGHT)
    env.initialize_subjects(num_subjects=SUBJECT_NUM)

    """MAIN ITERATION LOOP
    """
    for i in range(MAX_ITERATIONS):

        """PROCESS ENVIRONMENTAL CHANGES
        """
        pass

        """LOOP THROUGH CURRENT SUBJECTS
        """
        for subject, square in env.current_subject_dict.items():

            print(subject, square)



# RUN
if __name__ == "__main__":
    main()
