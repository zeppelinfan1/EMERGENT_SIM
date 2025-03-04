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
    lethal: bool = False # Determines if stepping on this terrain results in death
    movement_cost: int = 1 # Penalty for moving within terrain

# Types of terrains
LAND = TerrainType(name="LAND")
HOLE = TerrainType(name="HOLE", lethal=True, movement_cost=999)

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
class SpatialData:

    position: Position
    north: object = None
    south: object = None
    east: object = None
    west: object = None

    def __repr__(self):

        # Custom string representation to prevent infinite recursion
        return f"SpatialData(pos=({self.position.x}, {self.position.y}), " \
               f"N={self.north.spatial.position if self.north else None}, " \
               f"S={self.south.spatial.position if self.south else None}, " \
               f"E={self.east.spatial.position if self.east else None}, " \
               f"W={self.west.spatial.position if self.west else None})"

@dataclass
class Square:

    id: int # Unique identifier
    terrain: TerrainType # Type of terrain
    spatial: SpatialData # Square's position and neighboring squares (north, south, east, west)
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

    def step_on(self, character):

        if self.terrain.lethal:
            print(f"{character} fell into {self.terrain.name} (ID: {self.id}) and died!")
            return False
        print(f"{character} stepped on {self.terrain.name} (ID: {self.id}).")

        if self.objects:
            print(f"They see: {', '.join(obj.name for obj in self.objects)}.")

        return True  # Character survived

@dataclass
class Environment:

    width: int
    height: int
    default_terrain: float # How much of the terrain is land as opposed to alternatives (Holes)
    squares: list = field(default_factory=list)

    def __post_init__(self):

        square_map = {}
        # Start populating squares from the bottom right
        for y in range(self.height):

            for x in range(self.width):

                pos = Position(x, y)
                terrain_type = "LAND" if random.random() < self.default_terrain else "HOLE"
                new_square = Square(id=0, terrain=terrain_type, spatial=SpatialData(position=pos))
                self.squares.append(new_square)
                square_map[(x, y)] = new_square

        # Assign neighbors
        for square in self.squares:

            x, y = square.spatial.position.x, square.spatial.position.y
            square.spatial.north = square_map.get((x, y + 1))
            square.spatial.south = square_map.get((x, y - 1))
            square.spatial.west = square_map.get((x - 1, y))
            square.spatial.east = square_map.get((x + 1, y))

    def get_square(self, x, y):

        # Retrieve a square at the given (x, y) position.
        for square in self.squares:

            if square.spatial.position.x == x and square.spatial.position.y == y:

                return square

        return None  # Returns None if no square is found

    def get_random_square_subject(self):

        # Finds a random square that does not contain a subject
        empty_squares = [square for square in self.squares if square.subject is None]

        return random.choice(empty_squares) if empty_squares else None

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
    env.display()

