from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Entity:

    id: int = field(init=False)
    # Type
    entity_id: int = field(init=False)
    # Physical stats
    durability: float = 100.0
    parameters: dict = field(init=False)
    # State
    is_destroyed: bool = False
    # ID set
    _next_id: int = 0

    def __post_init__(self):

        Entity._next_id += 1
        self.id = Entity._next_id

        # Default parameters
        # Initialize neutral parameter dictionary (shared schema)
        self.parameters = {
            "strength": 0.0,
            "precision": 0.0,
            "stability": 0.0,
            "efficiency": 0.0,
            "aggression": 0.0,
            "edge_mean": 0.0,
            "edge_var": 0.0,
            "edge_max": 0.0,
            "area": 0.0,
        }


if __name__ == "__main__":
    e = Entity()