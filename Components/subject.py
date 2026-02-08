# IMPORTS
import numpy as np
import random
from dataclasses import dataclass, field
from collections import OrderedDict
import Components.network as nn
from Components.entity import Entity
from Components.mapping import Mapping
from Components.genetics import Genetics


@dataclass
class Memory:

    max_observations: int = 200
    memory: OrderedDict = field(default_factory=OrderedDict)

    def total_weight(self):

        return sum(self.memory.values())

    def add(self, embedding, label, increment=None, decay=0.99):

        # Loop through list of values
        for single_embedding, single_label in zip(embedding, label):

            key = tuple([float(single_label[0])] + list(single_embedding))

            if increment is None:
                increment = 1.0 / self.max_observations

            if key in self.memory:
                self.memory[key] += increment
            else:
                if self.total_weight() >= 1.0:

                    # Decay all weights to make room
                    for k in list(self.memory.keys()):

                        self.memory[k] *= decay

                    # Clean up very small entries
                    self.memory = OrderedDict((k, v) for k, v in self.memory.items() if v > 1e-6)

                self.memory[key] = increment

    def get_embeddings(self):

        embeddings = []
        labels = []

        for key, weight in self.memory.items():

            label = key[0]
            embedding = np.array(key[1:])

            # Convert weight into a number of samples
            count = int(round(weight * self.max_observations))

            for _ in range(count):

                embeddings.append(embedding)
                labels.append(label)

        return np.array(embeddings), np.array(labels)

@dataclass
class Subject(Entity):

    # Subject paramaters
    energy: int = 100

    # Genetics information
    gene_number: int = 3
    gene_length: int = 6
    genetics: Genetics = field(init=False) # Created in post init

    # Objective dict
    # objective_dict: dict = field(default_factory=dict)

    # Environmental related paramaters
    # perception_range: int = 2
    # env_memory: dict = field(default_factory=dict)

    # Features network paramaters
    # feature_embedding_length: int = 5 # Length of 3 for embeddings + 1 for numerous_features + 1 personal/observed = 5
    # feature_embeddings: dict = field(default_factory=dict)
    # feature_network: nn.Model = field(init=False)
    # feature_mapping: Mapping = field(init=False)
    # feature_memory: Memory = field(init=False)

    def __post_init__(self):
        
        # Entity initialization
        super().__post_init__()
        self.entity_id = 1 # Subject
        # Genetics
        self.genetics = Genetics(gene_number=self.gene_number, gene_length=self.gene_length)
        self.parameters = Genetics.interpret_genetics(self.genetics)

        # Network initialization
        # self.feature_network = self.initialize_network()
        # Mapping
        # self.feature_mapping = Mapping()
        # Memory
        # self.feature_memory = Memory()

    def generate_new_embedding(self, name, length=3):

        new_embedding = np.random.uniform(low=0.0, high=1.0, size=(length,))
        # Add to dict
        self.feature_embeddings[name] = new_embedding

    def initialize_network(self):

        # Instantiate the model
        network = nn.Model()

        # Add layers
        network.add(nn.Layer_Dense(self.feature_embedding_length, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
        network.add(nn.Activation_ReLU())
        network.add(nn.Layer_Dense(512, 512))
        network.add(nn.Activation_ReLU())
        network.add(nn.Layer_Dropout(rate=0.1))
        network.add(nn.Layer_Dense(512, 4))

        # Set loss, optimizer and accuracy objects
        network.set(
            loss=nn.Loss_Constrastive(),
            optimizer=nn.Optimizer_Adam(decay=1e-7),
            accuracy=nn.Accuracy_Constrastive()
        )

        network.finalize()

        return network

    def update_memory(self, env_section, verbage=False):

        new_keys = 0
        # Loop through squares and ensure that it is updated in the subjects enviromental memory
        for square_id, square_data in env_section.items():

            if square_id not in self.env_memory:
                self.env_memory[square_id] = square_data
                new_keys += 1
            else: self.env_memory[square_id] = square_data

        if new_keys > 0 and verbage: print(f"{new_keys} new keys added.")

if __name__ == "__main__":

    subject1 = Subject()
    print(subject1.parameters)
    subject2 = Subject()

    from Components.objects import Object
    object1 = Object()

    from Components.actions_original import Action
    Action.destory(subject1, subject2)

