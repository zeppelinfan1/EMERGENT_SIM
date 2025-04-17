# IMPORTS
import numpy as np
import random
from dataclasses import dataclass, field
from pyspark.sql import SparkSession
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
    current_subject_dict: dict = field(default_factory=dict)
    features: Features = field(init=False)
    square_map: dict = field(default_factory=dict)
    db: DB_API = field(default_factory=DB_API)
    movement_map = {
        0: (-1, -1), 1: (-1, 0), 2: (-1, 1),
        3: (0, -1), 4: (0, 0), 5: (0, 1),
        6: (1, -1), 7: (1, 0), 8: (1, 1)
    }

    def __post_init__(self):

        # Initialize features
        self.features = self.initialize_features()

        squares_upload = []
        square_features_upload = []
        # Start populating squares from the bottom right
        for y in range(self.height):

            for x in range(self.width):

                # Setting position based on width and height
                pos = Position(x, y)

                # Assign features based on random probability
                feature_list = self.assign_random_feature()

                # Create square
                new_square = Square(position=pos, features=feature_list)
                self.square_map[new_square.id] = new_square

                # Append for db upload
                squares_upload.append({
                    "id": new_square.id,
                    "x_coordinate": new_square.position.x,
                    "y_coordinate": new_square.position.y
                })

                for square_feature in new_square.features:

                    square_features_upload.append({
                        "square_id": new_square.id,
                        "feature_id": square_feature.id
                    })

        # Upload
        # spark_df = self.db.spark.createDataFrame(squares_upload)
        # self.db.insert_dataframe_spark(df=spark_df,table_name="squares")
        self.db.insert_mysql_bulk(table_name="squares", data=squares_upload)
        self.db.insert_mysql_bulk(table_name="square_features", data=square_features_upload)

    def initialize_features(self):

        # Gathering created features
        features = Features()
        feature_dict = features.feature

        """TEMPORARY - EVENTUALLY WILL GENERATE THESE EMERGENTLY
        """

        features_db_list = []
        for feat in [Feature(name="LAND", type="TERRAIN", energy_change=0, probability=1),
                     Feature(name="HOLE", type="TERRAIN", energy_change=-100, probability=0.05)]:

            # Already added to dictionary?
            if feat.type not in feature_dict.keys():
                # Create entry in value (list)
                feature_dict[feat.type] = []
                feature_dict[feat.type].append(feat)
            else:
                feature_dict[feat.type].append(feat)

            # Append for db bulk update
            features_db_list.append({
                "feature_name": feat.name,
                "feature_type": feat.type,
                "energy_change": feat.energy_change,
                "create_prob": feat.probability
            })

        # Upload to db
        self.db.insert_mysql_bulk(table_name="features", data=features_db_list)

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

    def get_square_from_subject(self, subject_id):

        return self.current_subject_dict[subject_id]

    def get_subject_from_square(self, square_id):

        pass

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

    def check_numerous_features(self, square):

        return [1] if len(square.features) > 1 else [0]

    def get_training_data(self, i, subject, current_square, observed_only=False):

        final_input_data = []
        final_target_data = []
        feature_memory = []
        """MAIN SQUARE i.e. OBSERVED
        """
        numerous = self.check_numerous_features(square=current_square)
        # Loop through features
        for feature in current_square.features:

            feature_key = f"F:{feature.id}"
            if feature_key not in subject.feature_embeddings:
                # Create unique embedding and add it to dict
                subject.generate_new_embedding(name=feature_key)

            # Concat numerous features and 'personal observation' values to list
            embedding = [float(x) for x in subject.feature_embeddings[feature_key]]
            input_data = embedding + numerous + [0]  # [0] signifies 'personally observed'

            # Check for observed energy change i.e. target value
            target_data = [(subject.energy_change + 100) / 200]

            # Append to final
            final_input_data.append(input_data)
            final_target_data.append(target_data)

            feature_memory.append({
                "iteration": i,
                "subject_id": subject.id,
                "target_subject_id": subject.id,
                "square_id": current_square.id,
                "feature_id": feature.id,
                "feature_embedding": ",".join(map(str, embedding)),
                "feature_label": float(target_data[0])
            })

        # If observed only, then return this, otherwise move on to perceived as well
        if observed_only: return final_input_data, final_target_data

        alternate_subject_squares = [square for square in subject.env_memory.values() if square.subject is not None
                                     and square.subject is not subject]  # Not the subject itself
        # Loop through env memory squares
        for alternate_subject_square in alternate_subject_squares:

            alternate_numerous = self.check_numerous_features(square=alternate_subject_square)
            # Keep track of the alternate subject
            alternate_subject = alternate_subject_square.subject

            # Loop through features within alternate square
            for alternate_feature in alternate_subject_square.features:

                # Ensure feature embedding exists
                alternate_feature_key = f"F:{alternate_feature.id}"
                if alternate_feature_key not in subject.feature_embeddings:
                    # Create unique embedding and add it to dict
                    subject.generate_new_embedding(name=alternate_feature_key)

                # Gather input/target data
                alternate_embedding = [float(x) for x in subject.feature_embeddings[alternate_feature_key]]
                alternate_input_data = alternate_embedding + alternate_numerous + [1]  # [1] signifies perceived

                # Check for observed energy change i.e. target value
                alternate_target_data = [(alternate_subject.energy_change + 100) / 200]

                # Append
                final_input_data.append(alternate_input_data)
                final_target_data.append(alternate_target_data)

                feature_memory.append({
                    "iteration": i,
                    "subject_id": subject.id,
                    "target_subject_id": alternate_subject.id,
                    "square_id": alternate_subject_square.id,
                    "feature_id": alternate_feature.id,
                    "feature_embedding": ",".join(map(str, alternate_embedding)),
                    "feature_label": float(alternate_target_data[0])
                })

        # Add to memory
        subject.feature_memory.add(embedding=final_input_data, label=final_target_data)
        # Update db
        self.db.insert_mysql_bulk(table_name="feature_memory", data=feature_memory)

    def check_is_within_bounds(self, x, y): # Needs to be altered to that subject can still move up/down if left/right unavailable

        # Check if (x, y) is within the grid boundaries.
        return 0 <= x < self.width and 0 <= y < self.height

    def add_subject(self, subject):

        square = self.get_random_square_subject()
        # Add subject to square
        square.subject = subject

    def initialize_subjects(self, num_subjects):

        subject_list = []
        for _ in range(num_subjects):

            new_subject = Subject(gene_number=6, gene_length=10, perception_range=2)
            new_square = self.get_random_square_subject()
            # Add to dict
            self.current_subject_dict[new_subject.id] = new_square.id
            # Add subject to square
            new_square.subject = new_subject
            subject_list.append({"id": new_subject.id})

        # Upload
        # spark_df = self.db.spark.createDataFrame(subject_list)
        # self.db.insert_dataframe_spark(df=spark_df, table_name="subjects")
        self.db.insert_mysql_bulk(table_name="subjects", data=subject_list)

    def update_energy_change(self, iteration, verbage=False):

        change_list = []
        for subject, square in self.current_subject_dict.items():

            real_square = self.square_map.get(square)
            # Real subject exists within square only
            real_subject = real_square.subject
            # Get square features
            features = real_square.features

            # Total sum of energy change
            total_energy_change = 0
            for feature in features:

                total_energy_change += feature.energy_change

                if total_energy_change != 0: # Add feature to this later
                    change_list.append({
                        "iteration": iteration,
                        "subject_id": real_subject.id,
                        "square_id": real_square.id,
                        "energy_change": feature.energy_change
                    })
                    if verbage:
                        print(f"Subject: {subject} experienced {total_energy_change} energy change in square {square}.")

            real_subject.energy_change = total_energy_change
            real_subject.energy += total_energy_change

        if len(change_list) > 0:
            # spark_df = self.db.spark.createDataFrame(change_list)
            # self.db.insert_dataframe_spark(spark_df, table_name="environmental_changes")
            self.db.insert_mysql_bulk(table_name="environmental_changes", data=change_list)

    def predict_square_energy_change(self, i, subject):

        prediction_d = {}
        prediction_list = []
        mapping_list = []
        memory_dict = subject.env_memory
        # Gather input data
        for square_id, square_value in memory_dict.items():

            numerous_features = self.check_numerous_features(square_value)
            for feature in square_value.features:

                # Gather embedding
                feature_key = f"F:{feature.id}"
                feature_embedding = subject.feature_embeddings.get(feature_key)
                if feature_embedding is None:  # Subject has no experience with it
                    # Generate and assign a new embedding for the unknown feature
                    feature_key = f"F:{feature.id}"
                    subject.generate_new_embedding(name=feature_key)
                    feature_embedding = subject.feature_embeddings[feature_key]

                square_input_data = list(feature_embedding) + numerous_features + [0]

                # Forward pass
                output = subject.feature_network.forward(X=np.array(square_input_data), training=None)

                # Update mapping
                if square_value.subject is subject:
                    label = (subject.energy_change + 100) / 200
                    subject.feature_mapping.update(output, label)

                    for val, embed in subject.feature_mapping.centroids.items():

                        mapping_list.append({
                            "iteration": i,
                            "subject_id": subject.id,
                            "label": val,
                            "embedding": ",".join(map(str, embed))
                        })

                # Gather heat map value
                mapping_pred = subject.feature_mapping.score(output)

                if not square_id in prediction_d.keys():
                    prediction_d[square_id] = float(mapping_pred)
                else:
                    prediction_d[square_id] += float(mapping_pred)

            prediction_list.append({
                "iteration": i,
                "subject_id": subject.id,
                "square_id": square_id,
                "prediction": prediction_d.get(square_id)
            })

        # Update db
        self.db.insert_mysql_bulk(table_name="square_prediction", data=prediction_list)
        self.db.insert_mysql_bulk(table_name="feature_mapping", data=mapping_list)

        return prediction_d

    def main_objective_choice(self, i, subject, prediction_d, env, scale=2.0) -> None:

        # Filter to only unoccupied squares
        available = {
            sid: score for sid, score in prediction_d.items()
            if env.square_map.get(sid) and env.square_map[sid].subject is None
        }

        if not available: return None

        # Convert to NumPy arrays
        square_ids = np.array(list(available.keys()))
        scores = np.array(list(available.values()))
        min_score = np.min(scores)
        max_score = np.max(scores)

        # Avoid division by zero if all scores are the same
        if max_score == min_score:
            probabilities = np.ones_like(scores) / len(scores)
        else:
            # Min-max normalization to range [0, 1]
            norm_scores = (scores - min_score) / (max_score - min_score)

            # Exaggerate differences with a temperature-controlled exponential
            exp_scores = np.exp(norm_scores * scale)
            probabilities = exp_scores / np.sum(exp_scores)

        # Add to db
        square_choice_list = []
        for square_id, prob in zip(square_ids, probabilities):

            square_choice_list.append({
                "iteration": i,
                "subject_id": int(subject.id),
                "square_id": int(square_id),
                "choice": float(prob)
            })

        self.db.insert_mysql_bulk(table_name="square_choice", data=square_choice_list)

        # Make random choice
        choice = np.random.choice(square_ids, p=probabilities)
        choice_prob = available.get(choice)

        # Compare to main objective
        if not subject.objective_dict: # If no existing objective
            subject.objective_dict[choice] = choice_prob
        elif choice_prob > max(subject.objective_dict.values()): # Update objective
            subject.objective_dict.clear()
            subject.objective_dict[choice] = choice_prob

    def next_square(self, current_square_id, final_square_id, pred_d):

        pass

    def find_path(self, subject, pred_d):

        # Find initial location
        initial_square_id = self.current_subject_dict.get(subject.id)
        # Find final destination
        final_square_id = max(subject.objective_dict, key=subject.objective_dict.get)

        # Calculate distance delta
        current_square_id = initial_square_id
        while current_square_id != final_square_id:

            self.next_square(current_square_id, final_square_id, pred_d)



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

    MAX_ITERATIONS = 100
    NUM_SUBJECTS = 10

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
            if total_energy_change != 0:
                print(f"Subject: {subject.id} experienced {total_energy_change} energy change.")

        for square in occupied_squares:

            subject = square.subject
            # print(f"Subject: {subject.id}")
            """PERCEIVING ENVIRONMENT
            """
            # Gather perception radius
            perceivable_env = env.get_squares_in_radius(square.position, subject.perception_range)
            # Update memory
            subject.update_memory(perceivable_env)

            """ FEATURE NEURAL NETWORK DATA GATHERING
            """
            final_input_data = []
            final_target_data = []
            # Check for newly encountered features and prep modular network if needed
            square_features = [feature for feature in square.features]

            # Check for presence of numerous_features
            numerous_features = [1] if len(square.features) > 1 else [0]

            for feature in square_features:

                feature_key = f"F:{feature.id}"
                if feature_key not in subject.feature_embeddings:
                    # Create unique embedding and add it to dict
                    subject.generate_new_embedding(name=feature_key)

                # Concat numerous features and 'personal observation' values to list
                embedding = [float(x) for x in subject.feature_embeddings[feature_key]]
                input_data = embedding + numerous_features + [0] # [0] signifies 'personally observed'

                # Check for observed energy change i.e. target value
                target_data = [(subject.energy_change + 100) / 200]

                # Append to final
                final_input_data.append(input_data)
                final_target_data.append(target_data)

            # Check for squares occupied by other subjects and prep modular network if needed
            alternate_subject_squares = [square for square in perceivable_env.values() if square.subject is not None
                                         and square.subject is not subject] # Not the subject itself

            for alternate_subject_square in alternate_subject_squares:

                # Keep track of the alternate subject
                alternate_subject = alternate_subject_square.subject

                # Check for presence of numerous_features
                alternate_numerous_features = [1] if len(alternate_subject_square.features) > 1 else [0]

                # Loop through features within alternate square
                for alternate_feature in alternate_subject_square.features:

                    # Ensure feature embedding exists
                    alternate_feature_key = f"F:{alternate_feature.id}"
                    if alternate_feature_key not in subject.feature_embeddings:
                        # Create unique embedding and add it to dict
                        subject.generate_new_embedding(name=alternate_feature_key)

                    # Gather input/target data
                    alternate_embedding = [float(x) for x in subject.feature_embeddings[alternate_feature_key]]
                    alternate_input_data = alternate_embedding + alternate_numerous_features + [1] # [1] signifies perceived

                    # Check for observed energy change i.e. target value
                    alternate_target_data = [(alternate_subject.energy_change + 100) / 200]

                    # Append
                    final_input_data.append(alternate_input_data)
                    final_target_data.append(alternate_target_data)

            # Add to memory
            subject.feature_memory.add(embedding=final_input_data, label=final_target_data)

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
            prediction_d = {}
            for subject_square_id, subject_square_value in subject.env_memory.items():

                numerous_features = [1] if len(subject_square_value.features) > 1 else [0]
                for feature in subject_square_value.features:

                    # Gather embedding
                    feature_key = f"F:{feature.id}"
                    feature_embedding = subject.feature_embeddings.get(feature_key)
                    if feature_embedding is None: # Subject has no experience with it
                        # Generate and assign a new embedding for the unknown feature
                        feature_key = f"F:{feature.id}"
                        subject.generate_new_embedding(name=feature_key)
                        feature_embedding = subject.feature_embeddings[feature_key]

                    square_input_data = list(feature_embedding) + numerous_features + [0]

                    # Forward pass
                    output = subject.feature_network.forward(X=np.array(square_input_data), training=None)

                    # Update mapping
                    if subject_square_value.subject is subject:
                        label = (subject.energy_change + 100) / 200
                        subject.feature_mapping.update(output, label)

                    # Gather heat map value
                    mapping_pred = subject.feature_mapping.score(output)

                    if not subject_square_id in prediction_d.keys():
                        prediction_d[subject_square_id] = mapping_pred
                    else:
                        prediction_d[subject_square_id] += float(mapping_pred)

            """SUBJECT ACTION
            """
            # Choose max value (or random if tied)
            max_value = max(prediction_d.values())
            best_keys = [k for k, v in prediction_d.items() if v == max_value]
            chosen_key = random.choice(best_keys)

            # Temporary - teleport to square instead of moving 1 square at a time
            new_square = env.square_map.get(chosen_key)
            if new_square.subject is not None: continue # No more than 1 subject per square
            square.subject = None
            new_square.subject = subject
            # print(f"Subject: {subject.id} moved from square {square.id} to {new_square.id}.")