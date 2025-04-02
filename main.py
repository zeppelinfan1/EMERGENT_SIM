# IMPORTS
from Components.environment import Environment
from Components.db_api import DB_API

MAX_ITERATIONS = 100
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
        env.update_energy_change(i, verbage=True)

        """LOOP THROUGH CURRENT SUBJECTS - SUBJECT DECISION MAKING PROCESS
        """
        current_positions = []
        for subject_id, square_id in env.current_subject_dict.items():

            square = env.square_map.get(square_id)
            subject = square.subject
            current_positions.append({
                "iteration": i,
                "subject_id": subject_id,
                "square_id": square_id,
                "square_x": square.position.x,
                "square_y": square.position.y
            })

            """GATHERING ENVIRONMENTAL TRAINING DATA
            """
            # Perceiving environment in surrounding radius
            perceivable_env = env.get_squares_in_radius(square.position, subject.perception_range)
            # Update memory
            subject.update_memory(perceivable_env)

            # Gather feature training data
            env.get_training_data(subject, square)

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
            prediction_d = env.predict_square_energy_change(subject)
            chosen_id = env.choose_square(prediction_d, env)
            if chosen_id is None: continue  # Skip turn if no unoccupied squares available

            # Process new square movement
            new_square = env.square_map[chosen_id]
            square.subject = None
            new_square.subject = subject
            env.current_subject_dict[subject.id] = chosen_id
            # print(f"Subject: {subject.id} moved from square {square.id} to {new_square.id}.")

        # Update database
        env.db.insert_mysql_bulk(table_name="current_positions", data=current_positions)





# RUN
if __name__ == "__main__":
    main()
