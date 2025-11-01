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
import random
from dataclasses import dataclass, field
from Components.entity import Entity




@dataclass
class Object(Entity):

    def __post_init__(self):

        # Entity initialization
        super().__post_init__()
        self.entity_id = 2 # Object
        # Parameters
        self.parameters = self.generate_parameters()

    def generate_parameters(self) -> dict:

        parameters = {
            "strength": round(random.uniform(0.3, 1.0), 3),  # how hard the material is
            "stability": round(random.uniform(0.2, 1.0), 3),  # internal cohesion
            "edge_mean": round(random.uniform(0.5, 2.0), 3),  # average uniformity
            "edge_var": round(random.uniform(0.0, 0.5), 3),  # irregularity (0=perfect)
            "edge_max": round(random.uniform(1.0, 3.0), 3),  # maximum local defect
            "area": round(random.uniform(0.5, 2.0), 3),  # size / contact area
        }

        return parameters

if __name__ == "__main__":
    from Components.subject import Subject
    subject1 = Subject(gene_number=6, gene_length=10)
    object1 = Object()

