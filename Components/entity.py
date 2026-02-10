from dataclasses import dataclass, field
from typing import Tuple
from Components.states import State

@dataclass
class Entity:

    id: int = field(init=False)
    # Type
    entity_id: int = field(init=False)
    # State
    state: State = field(init=False)
    # ID set
    _next_id: int = 0

    def __post_init__(self):

        Entity._next_id += 1
        self.id = Entity._next_id

        self.state = State()



if __name__ == "__main__":
    e = Entity()