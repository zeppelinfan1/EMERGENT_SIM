"""
ECOSYSTEM EVOLUTION SIMULATOR
"""
MAX_ITERATIONS = 1
ENV_HEIGHT = 50
ENV_WIDTH = 100
SUBJECT_NUM = 2
SUBJECT_GENE_NUM = 8
SUBJECT_GENE_LEN = 10


# IMPORTS
from Components.environment import Environment
from Components.subject import Subject


# MAIN OBJECT
def main():

    # Initialize Environment
    env = Environment(height=ENV_HEIGHT, width=ENV_HEIGHT, default_terrain=0.97)

    # Populate subject into empty square
    subject_d = {}
    for _ in range(SUBJECT_NUM):

        subject = Subject(gene_number=SUBJECT_GENE_NUM, gene_length=SUBJECT_GENE_LEN)
        env.add_subject(subject)

    # Start iterations
    for i in range(MAX_ITERATIONS):

        # Get squares with active subjects
        occupied_squares = env.get_occupied_squares()
        print(f"Iteration Number: {i}")
        # For each subject
        for occupied_square in occupied_squares:

            square = occupied_square.spatial
            print(square)

            # Prepare perception training input
            pass

            # Train
            pass

            # Peform action
            pass

# RUN
if __name__ == "__main__":
    main()
