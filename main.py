# IMPORTS
import pandas as pd, numpy as np
from Components.environment import Environment
from Components.subject import Subject
from Components.network import Model
from Components.db_api import DB_API

MAX_ITERATIONS = 1
ENV_HEIGHT = 50
ENV_WIDTH = 20
SUBJECT_NUM = 10
SUBJECT_GENE_NUM = 8
SUBJECT_GENE_LEN = 10
SUBJECT_PERCEPTION_RANGE = 3

# MAIN OBJECT
def main():

    """INITIALIZE ENVIRONMENT AND SUBJECTS
    """
    env = Environment(width=ENV_WIDTH, height=ENV_HEIGHT)
    env.initialize_subjects(num_subjects=SUBJECT_NUM)

    """MAIN ITERATION LOOP
    """
    for i in range(MAX_ITERATIONS):

        print(f"[~] Iteration #: {i}" + "\n" + "-" * 20)
        """PROCESS ENVIRONMENTAL CHANGES
        """
        env.update_energy_change(verbage=True)

        """LOOP THROUGH CURRENT SUBJECTS
        """
        for subject_id, square_id in env.current_subject_dict.items():

            square = env.square_map.get(square_id)
            subject = square.subject

            """GATHERING ENVIRONMENTAL TRAINING DATA
            """
            # Perceiving environment in surrounding radius
            perceivable_env = env.get_squares_in_radius(square.position, subject.perception_range)
            # Update memory
            subject.update_memory(perceivable_env)

            # Gather feature training data
            input_data, target_values = env.get_training_data(subject, square)

            """ FEATURE NEURAL NETWORK DATA TRAINING
            """
            # Gather full training data from memory
            X, y = subject.feature_memory.get_embeddings()
            # Constrastive pairs
            pairs, pair_labels = subject.feature_network.generate_contrastive_pairs(X, y)
            # Train
            subject.feature_network.train(X=pairs, y=pair_labels, epochs=1, batch_size=128)

            """SQUARE PREDICTION
            """
            prediction_d = env.predict_square_energy_change()





# RUN
if __name__ == "__main__":
    main()
