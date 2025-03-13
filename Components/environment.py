# IMPORTS
import numpy as np
import random
from dataclasses import dataclass, field
from Components.subject import Subject
from Components.db_api import DB_API

# DATACLASSES
@dataclass
class Terrain:

    name: str
    energy_penalty: int
    probability: float = 0
    overall_probability: float = field(default_factory=float)

# Types of terrains
LAND = Terrain(name="LAND", energy_penalty=0, probability=1)
HOLE = Terrain(name="HOLE", energy_penalty=100, probability=0.03)

@dataclass
class Object:

    name: str
    energy_penalty: int
    probability: float = 0
    overall_probability: float = field(default_factory=float)

@dataclass
class Features:

    feature: dict = field(default_factory=dict) # Either: Terrain, Object, ... potentially more later


@dataclass
class Position:

    x: int
    y: int

@dataclass
class Square:

    id: int # Unique identifier
    position: Position # x, y coordinates
    features: list = field(default_factory=list)
    subject: Subject = None
    # For assigning id - will increment
    last_id = 0

    def __post_init__(self):

        Square.last_id += 1
        self.id = Square.last_id # Assigns unique value

    def add_object(self, obj):

        self.objects.append(obj)

    def remove_object(self, obj):

        if obj in self.objects:
            self.objects.remove(obj)

@dataclass
class Environment:

    width: int
    height: int
    features_list: list
    square_map: dict = field(default_factory=dict)
    movement_map = {
        0: (-1, -1), 1: (-1, 0), 2: (-1, 1),
        3: (0, -1), 4: (0, 0), 5: (0, 1),
        6: (1, -1), 7: (1, 0), 8: (1, 1)
    }

    def __post_init__(self):

        # Initialize features
        features = Features()
        feature_id = 1 # Will increment
        for var_name, var_value in globals().items():

            if isinstance(var_value, tuple(self.features_list)):
                features.feature[feature_id] = var_value
                feature_id += 1

        print(features)

        # Start populating squares from the bottom right
        for y in range(self.height):

            for x in range(self.width):

                # Setting position based on width and height
                pos = Position(x, y)

                # Assign features based on random probability
                for feature_type in features.feature.values():

                    pass

                # Create square
                new_square = Square(id=0, position=pos)
                self.square_map[(pos.x, pos.y)] = new_square

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

                new_position = (position.x + dx, position.y + dy)  # ✅ Tuple format
                square = self.square_map.get(new_position)  # ✅ Lookup using tuple
                if square: neighbors.append(square)

        return neighbors

    def get_training_data(self, neighbors: list, subject_position):

        input_data = [] # Features of environment squares
        target_data = [] # Survival criteria

        for square in neighbors:

            terrain_encoding = [1, 0] if square.terrain == "LAND" else [0, 1] # One hot encoding for terrain - temporary
            object_presence = [1] if square.objects else [0]
            subject_presence = [1] if square.subject else [0]

            # Find relative position
            dx = square.position.x - subject_position.x
            dy = square.position.y - subject_position.y
            move_index = next((key for key, value in self.movement_map.items() if value == (dx, dy)), None)
            # Target data append
            if move_index is not None:
                target_data.append(move_index)

            # Combine into one row
            square_features = terrain_encoding + object_presence + subject_presence
            input_data.append(square_features)

        # Convert into array
        data_array = np.array(input_data)
        target_array = np.array(target_data)

        return data_array, target_array

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
                row.append("H" if square and square.terrain == "HOLE" else "L")  # H = Hole, L = Land

            print(" ".join(row))


if __name__ == "__main__":

    env = Environment(width=1, height=1, features_list=[Terrain, Object])
    env.add_subject(Subject(5, 10, 9, 2))
    occupied_squares = env.get_occupied_squares()

    for square in occupied_squares:

        subject = square.subject
        # Gather perception radius
        perceivable_env = env.get_squares_in_radius(square.position, subject.perception_range)
        # Update memory
        subject.update_memory(perceivable_env)



