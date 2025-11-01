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
    is_desroyed: bool = False
    # ID set
    _next_id: int = 0

    def __post_init__(self):

        Entity._next_id += 1
        self.id = Entity._next_id


if __name__ == "__main__":
    e = Entity()