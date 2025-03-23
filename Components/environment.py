# IMPORTS
import numpy as np
import random
from dataclasses import dataclass, field
from Components.subject import Subject
from Components.db_api import DB_API

# DATACLASSES
@dataclass
class Feature:

    id: int = field(init=False)
    name: str
    type: str
    energy_change: int
    probability: float = 0
    overall_probability: float = field(default_factory=float)

    # For assigning id - will increment
    last_id = 0

    def __post_init__(self):

        Feature.last_id += 1
        self.id = Feature.last_id

# Types of terrains
LAND = Feature(name="LAND", type="TERRAIN", energy_change=0, probability=1)
HOLE = Feature(name="HOLE", type="TERRAIN", energy_change=-100, probability=0.05)

@dataclass
class Features:

    feature: dict = field(default_factory=dict) # Either: Terrain, Object, ... potentially more later


@dataclass
class Position:

    x: int
    y: int

@dataclass
class Square:

    id: int = field(init=False)
    position: Position # x, y coordinates
    features: list = field(default_factory=list)
    subject: Subject = None

    # For assigning id - will increment
    last_id = 0

    def __post_init__(self):

        Square.last_id += 1
        self.id = Square.last_id # Assigns unique value

    def add_feature(self, feature):

        self.features.append(feature)

    def remove_feature(self, feature):

        if feature in self.features:
            self.features.remove(feature)

@dataclass
class Environment:

    width: int
    height: int
    features: Features = field(init=False)
    square_map: dict = field(default_factory=dict)
    movement_map = {
        0: (-1, -1), 1: (-1, 0), 2: (-1, 1),
        3: (0, -1), 4: (0, 0), 5: (0, 1),
        6: (1, -1), 7: (1, 0), 8: (1, 1)
    }

    def __post_init__(self):

        # Initialize features
        self.features = self.initialize_features()

        # Start populating squares from the bottom right
        for y in range(self.height):

            for x in range(self.width):

                # Setting position based on width and height
                pos = Position(x, y)

                # Assign features based on random probability
                feature_list = self.assign_random_feature()

                # Create square
                new_square = Square(position=pos, features=feature_list)
                self.square_map[(pos.x, pos.y)] = new_square

    def initialize_features(self):

        # Gathering created features
        features = Features()
        feature_dict = features.feature
        for var_name, var_value in globals().items():

            # If the value is a Feature object
            if isinstance(var_value, Feature):

                # Already added to dictionary?
                if var_value.type not in feature_dict.keys():
                    # Create entry in value (list)
                    feature_dict[var_value.type] = []
                    feature_dict[var_value.type].append(var_value)
                else:
                    feature_dict[var_value.type].append(var_value)

        # Loop through keys
        for feature_type in feature_dict.keys():

            # Gather values and loop to sum up total probability
            value_list = feature_dict[feature_type]
            total_probability = 0
            for obs in value_list:

                total_probability += obs.probability

            # Calculate individual probability
            for obs in value_list:

                obs.overall_probability = obs.probability / total_probability

        return features

    def assign_random_feature(self) -> list:

        # Loop through keys:
        feature_selection = []
        for feature_type, feature_list in self.features.feature.items():

            choices = feature_list
            probabilities = [feature.overall_probability for feature in feature_list]
            # Select a feature randomly based on probabilities
            selected_feature = random.choices(choices, weights=probabilities, k=1)[0]
            feature_selection.append(selected_feature)

        return feature_selection

    def get_square(self, x, y):

        # Retrieve a square at the given (x, y) position.
        for square in self.square_map.values():

            if square.position.x == x and square.position.y == y:

                return square

        return None  # Returns None if no square is found

    def get_squares_in_radius(self, pos, radius):

        x, y = pos.x, pos.y
        env_section = {}

        # Loop over entire radius - x and y
        for dx in range(-radius, radius + 1):

            for dy in range(-radius, radius + 1):

                new_x, new_y = x + dx, y + dy

                # Ensure that square is within total environment
                if self.check_is_within_bounds(new_x, new_y):
                    # Add square to dict
                    square_data = self.get_square(new_x, new_y)
                    env_section[square_data.id] = square_data

        return env_section


    def get_movement_delta(self, move_index: int):

        # Retrieve (dx, dy) movement from index
        return self.movement_map.get(move_index, (0, 0))  # Default to no movement

    def get_random_square_subject(self):

        # Finds a random square that does not contain a subject
        empty_squares = [square for square in self.square_map.values() if square.subject is None]

        return random.choice(empty_squares) if empty_squares else None

    def get_occupied_squares(self) -> list:

        # Returns a list of tuples (subject, square) for all subjects in the environment
        return [square for square in self.square_map.values() if square.subject is not None]

    def get_neighbors(self, position, perception_range) -> list:

        neighbors = []
        for dx in range(-perception_range, perception_range + 1):

            for dy in range(-perception_range, perception_range + 1):

                new_position = (position.x + dx, position.y + dy)  # Tuple format
                square = self.square_map.get(new_position)  # Lookup using tuple
                if square: neighbors.append(square)

        return neighbors

    def get_training_data(self, env_memory: dict, feature_memory: dict):

        input_data = [] # 1 hot for features for each square in environment
        target_data = [] # Based on feature memory

        # Loop through squares
        for square in env_memory.values():

            # Loop through features
            for feature in square.features:

                # One hot encoding i.e. input data
                one_hot_feature = [1 if feature.name == mem_name else 0 for mem_name in feature_memory.keys()]
                input_data.append(one_hot_feature)
                # Target value from feature memory
                target_value = feature_memory.get(feature.name)
                target_data.append(target_value)

        return np.array(input_data), np.array(target_data)

    def check_is_within_bounds(self, x, y): # Needs to be altered to that subject can still move up/down if left/right unavailable

        # Check if (x, y) is within the grid boundaries.
        return 0 <= x < self.width and 0 <= y < self.height

    def add_subject(self, subject):

        square = self.get_random_square_subject()
        # Add subject to square
        square.subject = subject


    def display(self):

        # Prints the environment, showing 'H' for holes and 'L' for land
        for y in reversed(range(self.height)):  # Print from top to bottom

            row = []
            for x in range(self.width):

                square = self.get_square(x, y)
                if square.features[0].name == "LAND": row.append("L")
                else: row.append("H")

            print(" ".join(row))


if __name__ == "__main__":

    MAX_ITERATIONS = 1
    NUM_SUBJECTS = 1

    env = Environment(width=50, height=20)

    for _ in range(NUM_SUBJECTS):

        env.add_subject(Subject(gene_number=6, gene_length=10, perception_range=2))

    for i in range(MAX_ITERATIONS):

        print(f"Iteration Number: {i}")
        occupied_squares = env.get_occupied_squares()

        for square in occupied_squares:

            subject = square.subject
            """PROCESS ENVIRONMENTAL FEATURES
            """
            # I.e. energy change for subjects
            total_energy_change = sum([feature.energy_change for feature in square.features])
            subject.energy_change = total_energy_change
            subject.energy += total_energy_change

        for square in occupied_squares:

            subject = square.subject
            print(f"Subject: {subject.id}")
            """PERCEIVING ENVIRONMENT
            """
            # Gather perception radius
            perceivable_env = env.get_squares_in_radius(square.position, subject.perception_range)
            # Update memory
            subject.update_memory(perceivable_env)

            """ FEATURE NEURAL NETWORK RETRAINING
            """
            # Check for newly encountered features and prep modular network if needed
            square_features = [feature for feature in square.features]

            for feature in square_features:

                feature_key = f"F:{feature.id}"
                if feature_key not in subject.feature_embeddings:
                    # Create unique embedding and add it to dict
                    subject.generate_new_embedding(name=feature_key)

                # Check for presence of numerous_features
                numerous_features = [1] if len(square.features) > 1 else [0]

                # Concat numerous features value to list
                embedding = [float(x) for x in subject.feature_embeddings[feature_key]]
                input_data = embedding + numerous_features

                # Check for observed energy change i.e. target value
                target_data = [(subject.energy_change + 100) / 200]

                # Generate constrastive pairs










            #
            #     input_data = []
            #     target_data = []
            #     # Check for presence of another feature - input data value
            #     numerous_features = 1 if len(square.features) > 1 else 0
            #     input_data.append([numerous_features])
            #     # Check for energy change - target data value
            #     energy_change = (subject.energy_change + 100) / 200
            #     target_data.append([energy_change])
            #
            #     # Train network
            #     subject.modular_networks[feature_key].train(X=np.array(input_data), y=np.array(target_data), epochs=10, batch_size=128)
            #
            # # Check for squares occupied by other subjects and prep modular network if needed
            # env_subject_squares = [square for square in perceivable_env.values() if square.subject is not None
            #                        and square.subject is not subject] # Not the subject itself
            # input_data = []
            # target_data = []
            # for alt_square in env_subject_squares:
            #
            #     alt_subject = alt_square.subject
            #     alt_subject_key = f"SUBJECT:{alt_subject.id}"
            #     if alt_subject_key not in subject.modular_networks.keys():
            #         # So far only 1 input array for features networks with 1 element [numerous_features]
            #         subject.modular_networks[alt_subject_key] = subject.initialize_network(input_features=1)
            #
            #     # Gather training/target data for alternate subjects i.e. sensory network
            #     # Check for presence of another feature - input data value
            #     alt_numerous_features = 1 if len(alt_square.features) > 1 else 0
            #     input_data.append([alt_numerous_features])
            #     # Check for energy change - target data value
            #     energy_change = (alt_subject.energy_change + 100) / 200
            #     target_data.append([energy_change])
            #
            #     subject.modular_networks[alt_subject_key].train(X=np.array(input_data), y=np.array(target_data), epochs=10,
            #                                             batch_size=128)
            #
            # # Forward pass through modular networks in order to get Attention Mechanism training data
            # for current_feature in square.features:
            #
            #     output = subject.modular_networks[f"FEATURE:{current_feature.id}"].forward(X=np.array(numerous_features), training=None)
            #     # subject.modular_networks["ATTN"].train(X=np.array())
            #
            # """PLAN ASSESSMENT
            # """
            # # Loop through each square in subject memory
            # for square_id, square_data in subject.env_memory.items():
            #
            #     # Gather input data
            #     pass
