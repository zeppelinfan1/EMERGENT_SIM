"""
OBJECT FUNCTIONALITY
MAPPING - X and Y coordinates
X coordinate determines energy drain (negative) / provide (positive)
Y coordinate determines object destroy (negative) / create (positive)
Example:
(-1, -1) Drain and Destroy - Consumes a lot of energy and destroys an object
(-1, 1) Drain and Create - Consumes a lot of energy and creates an object
(1, -1) Provide and Destroy - Provides energy and destroys an object
(1, 1) Provide and Create - Provides energy and creates an object
"""

import numpy as np
import random
from dataclasses import dataclass, field




@dataclass
class Object:

    id: int = field(init=False)
    fx: float = None
    fy: float = None
    sx: float = None
    sy: float = None
    durability: float = 1.0
    destroyed: bool = False

    # For assigning id - will increment
    last_subject = 0

    def __post_init__(self):

        Object.last_subject += 1
        self.id = Object.last_subject

        # Functional mapping: energy drain or provide (x) vs create or destory (y)
        if self.fx is None or self.fy is None:
            self.fx, self.fy = (random.random() * 2 - 1), (random.random() * 2 - 1)
        self.functional_map = (self.fx, self.fy)
        self.fx, self.fy = self.functional_map

        # Structural mapping: durability (x) vs cost (y)
        if self.sx is None or self.sy is None:
            self.sx, self.sy = (random.random() * 2 - 1), (random.random() * 2 - 1)
        self.structural_map = (self.sx, self.sy)
        self.sx, self.sy = self.structural_map

    def destroy(self, actor, target):

        create_power = abs(self.fy)
        energy_cost = create_power * (1 + max(0., self.sy))

        actor.energy -= energy_cost

        resistance = 1 - target.sx
        target.durability -= create_power * resistance

        if target.durability <= 0:
            target.destroyed = True
            print(f"{actor.id} destroyed {target.id}")
        else:
            print(f"{actor.id} damaged {target.id}. Durability now {target.durability:.2f}")


if __name__ == "__main__":
    from Components.subject import Subject
    subject1 = Subject(gene_number=6, gene_length=10, perception_range=2)
    rock = Object(fy=-0.8, sx=0.5, sy=0.2)  # Moderate destroy ability, decent durability
    tree = Object(sx=0.2, sy=-0.3 ) # Low durability

    # simulate destruction
    rock.destroy(actor=subject1, target=tree)

    print(f"Agent energy: {subject1.energy:.2f}")
    print(f"Tree destroyed? {tree.destroyed}")
