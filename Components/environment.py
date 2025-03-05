"""
2D Space for individuals within the population to interact with
various environmental features.

Step 1) Setup environment and 1st feature (holes)
"""

# IMPORTS
import numpy as np
import random
from dataclasses import dataclass, field
from Components.subject import Subject
from Components.db_api import db_api

# DATACLASSES
@dataclass
class TerrainType:

    name: str
    energy_penalty: int = 0

# Types of terrains
LAND = TerrainType(name="LAND")
HOLE = TerrainType(name="HOLE", energy_penalty=100)

@dataclass
class Object:

    name: str
    interactable: bool = True # May pick up, drop or use the object
    weight: int = 1

@dataclass
class Position:

    x: int
    y: int

@dataclass
class Square:

    id: int # Unique identifier
    terrain: TerrainType # Type of terrain
    position: Position # x, y coordinates
    objects: list = field(default_factory=list) # Objects within square
    subject: Subject = None

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
    default_terrain: float # How much of the terrain is land as opposed to alternatives (Holes)
    square_map: dict = field(default_factory=dict)

    def __post_init__(self):

        # Start populating squares from the bottom right
        for y in range(self.height):

            for x in range(self.width):

                pos = Position(x, y)
                terrain_type = "LAND" if random.random() < self.default_terrain else "HOLE"
                new_square = Square(id=0, terrain=terrain_type, position=pos)
                self.square_map[(pos.x, pos.y)] = new_square

    def get_square(self, x, y):

        # Retrieve a square at the given (x, y) position.
        for square in self.square_map.values():

            if square.position.x == x and square.position.y == y:

                return square

        return None  # Returns None if no square is found

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

    def get_square_input_data(self, neighbors: list):

        input_data = []

        for square in neighbors:

            terrain_encoding = [1, 0] if square.terrain == "LAND" else [0, 1] # One hot encoding for terrain - temporary
            object_presence = [1] if square.objects else [0]
            subject_presence = [1] if square.subject else [0]

            # Combine into one row
            square_features = terrain_encoding + object_presence + subject_presence
            input_data.append(square_features)

        # Convert into array
        data_array = np.array(input_data)

        return data_array

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
    env = Environment(25, 10, default_terrain=0.97)
    env.add_subject(Subject(5, 10))
    occupied_squares = env.get_occupied_squares()
    # For each subject
    for occupied_square in occupied_squares:
        square = occupied_square.position
        # Prepare perception training input
        neighboring_squares = env.get_neighbors(position=square, perception_range=1)  # Also includes square itself
        neighbor_input_data = env.get_square_input_data(neighboring_squares)

        # Train
        pass

        # Peform action
        pass

