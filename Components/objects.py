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
        if self.fx is None:
            self.fx = random.random() * 2 - 1
        if self.fy is None:
            self.fy = random.random() * 2 - 1
        self.functional_map = (self.fx, self.fy)

        # Structural mapping: durability (x) vs cost (y)
        if self.sx is None:
            self.sx = random.random() * 2 - 1
        if self.sy is None:
            self.sy = random.random() * 2 - 1
        self.structural_map = (self.sx, self.sy)

    def destroy(self, actor, target):

        power = abs(self.fy)
        energy_cost = (
                abs(self.fy) * (1.0 + max(0.0, self.sy))  # cost scales with destruction and cost axis
                - self.fx  # offset by object’s energy contribution
        )

        actor.energy -= energy_cost

        resistance = 1 - target.sx
        target.durability -= power * resistance

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
