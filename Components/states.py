"""
STATE
The current state of the entity
"""

# IMPORTS
import numpy as np
import random, math
from dataclasses import dataclass, field



# OBJECTS
@dataclass
class State:

    temp: int = 1