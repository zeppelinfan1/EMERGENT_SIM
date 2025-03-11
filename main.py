"""
ECOSYSTEM EVOLUTION SIMULATOR
"""

MAX_ITERATIONS = 30
ENV_HEIGHT = 50
ENV_WIDTH = 100
SUBJECT_NUM = 1
SUBJECT_GENE_NUM = 8
SUBJECT_GENE_LEN = 10
SUBJECT_PERCEPTION_RANGE = 1
NN_FEATURE_COUNT = 4 # Terrain Land, Terrain Hole, Object Presence, Subject Presence


# IMPORTS
import pandas as pd, numpy as np
from Components.environment import Environment
from Components.subject import Subject
from Components.nn_brain import Model
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
                          features=NN_FEATURE_COUNT)
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
            input_data, target_data = env.get_training_data(neighboring_squares, square)
            # Append new historical data and then query to get final training data df
            db.insert_hist(subject_id=subject.id, square_id=occupied_square.id, feature_count=NN_FEATURE_COUNT,
                           feature_data=input_data, target_data=target_data)
            training_df = db.get_hist(subject_id=subject.id)

            # Train
            X, y = training_df.iloc[:, 3:-1], training_df.iloc[:, -1]
            X = np.array(X, dtype=int)
            y = np.array(y, dtype=int)
            subject.brain.train(X, y, epochs=10, batch_size=128)

            # Peform action
            softmax_output = subject.brain.forward(X=X, training=None)
            best_move_index = np.argmax(softmax_output, axis=1)[0]  # Select highest probability move

            # Convert index to movement delta
            dx, dy = env.get_movement_delta(best_move_index)
            print(f"({dx}, {dy})")

            # Get subject's current position
            new_x = occupied_square.position.x + dx
            new_y = occupied_square.position.y + dy

            # Ensure the move is within bounds
            if env.check_is_within_bounds(new_x, new_y):
                # Find the new square
                new_square = env.get_square(new_x, new_y)
                # Ensure the square is not occupied before moving
                if new_square and new_square.subject is None:
                    # Move subject to new square
                    occupied_square.subject = None  # Remove subject from old square
                    new_square.subject = subject  # Assign subject to new square
                    # print(f"Subject moved to ({new_x}, {new_y}).")
                else:
                    print(f"Square ({new_x}, {new_y}) is occupied, staying in place.")
            else:
                print(f"Move out of bounds: ({new_x}, {new_y}) - Staying in place.")


# RUN
if __name__ == "__main__":
    main()
