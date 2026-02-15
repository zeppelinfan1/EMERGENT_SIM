"""
OBJECT FUNCTIONALITY
Symbol	Domain	Function	Interpretation
fx	Functional (X-axis)	Energy flow	Positive = provides energy to the actor; negative = drains actor energy.
    Controls short-term energetic effect of interactions (fuel vs fatigue).
fy	Functional (Y-axis)	Creation vs destruction	Positive = creates or constructs; negative = destroys or damages.
    Determines whether an interaction builds or breaks entities.
sx	Structural (X-axis)	Durability / resilience	Positive = resistant, hard to damage; negative = fragile, breaks easily.
    Affects how much damage an object can take and how well it resists destruction.
sy	Structural (Y-axis)	Production cost / complexity	Positive = expensive or energy-intensive to create or use; negative = cheap or efficient.
    Modulates the energy cost of actions and potential trade value.
"""

import numpy as np
import random, math
from dataclasses import dataclass, field
from Components.entity import Entity




@dataclass
class Object(Entity):

    def __post_init__(self):

        # Entity initialization
        super().__post_init__()
        self.entity_id = 2 # Object
        # Parameters
        self.projection_n = 1

    def generate_projections(self, max_dir=5.0, max_z=8.0):

        projection_dict = {}
        for i in range(self.projection_n):

            theta = random.uniform(0, 2 * math.pi)
            r = random.uniform(0.2, 1.0) * max_dir
            x = math.cos(theta) * r
            y = math.sin(theta) * r
            z = random.uniform(-max_z, max_z)

            # Putting each segment through the alogorithm
            projection_dict[i] = {
                "x": x,  # X-coordinate
                "y": y,  # y-coordinate
                "z": z,  # Magnitude
            }

        return projection_dict

if __name__ == "__main__":
    object1 = Object()
    print(object1)

