"""
ECOSYSTEM EVOLUTION SIMULATOR
"""
MAX_ITERATIONS = 1
ENV_HEIGHT = 50
ENV_WIDTH = 100
SUBJECT_NUM = 2
SUBJECT_GENE_NUM = 8
SUBJECT_GENE_LEN = 10
SUBJECT_PERCEPTION_RANGE = 1
NN_FEATURE_COUNT = 4 # Terrain Land, Terrain Hole, Object Presence, Subject Presence


# IMPORTS
import pandas as pd
from Components.environment import Environment
from Components.subject import Subject
from Components.db_api import DB_API


# MAIN OBJECT
def main():

    # Initialize Environment
    env = Environment(height=ENV_HEIGHT, width=ENV_HEIGHT, default_terrain=0.97)
    # Database creation and main tables
    db = DB_API()
    db.create_hist_table(inputs=NN_FEATURE_COUNT)

    # Populate subject into empty square
    subject_d = {}
    for _ in range(SUBJECT_NUM):

        subject = Subject(gene_number=SUBJECT_GENE_NUM,
                          gene_length=SUBJECT_GENE_LEN,
                          perception=SUBJECT_PERCEPTION_RANGE)
        env.add_subject(subject)

    # Start iterations
    for i in range(MAX_ITERATIONS):

        # Get squares with active subjects
        occupied_squares = env.get_occupied_squares()
        print(f"Iteration Number: {i}")
        # For each subject
        for occupied_square in occupied_squares:

            square = occupied_square.position
            subject = occupied_square.subject
            # Prepare perception training input
            neighboring_squares = env.get_neighbors(position=square, perception_range=SUBJECT_PERCEPTION_RANGE) # Also includes square itself
            input_data, target_data = env.get_training_data(neighboring_squares)
            # Query historical training data, append new
            hist_df = db.get_hist(subject_id=subject.id)
            db.insert_hist(subject_id=subject.id, square_id=occupied_square.id, feature_count=NN_FEATURE_COUNT,
                           feature_data=input_data, target_data=target_data)

            print(db.get_hist(subject_id=subject.id))
            # Combine
            pass

            # Train
            pass

            # Peform action
            pass

# RUN
if __name__ == "__main__":
    main()
