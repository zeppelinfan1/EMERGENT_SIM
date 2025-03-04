import numpy as np
import random
from dataclasses import dataclass, field




@dataclass
class Object:

    mapping: (float, float) = field(init=False)

    def __post_init__(self):

        # Random x, y pairing between -1 and 1
        self.mapping = (random.random() * 2) - 1, (random.random() * 2) - 1

        print(self.mapping)




obj1 = Object()