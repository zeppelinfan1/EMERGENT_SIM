import numpy as np

class Mapping:

    def __init__(self, embedding_dim=4, decay_rate=0.95):

        self.embedding_dim = embedding_dim
        self.decay_rate = decay_rate

        # Storing centroid points
        self.centroids = {}
        # For tracking number of updates
        self.counts = {}


    def update(self, embedding, label):

        label = float(label)
        # Create entry for centroid in dict if not already exists
        if label not in self.centroids:
            self.centroids[label] = np.zeros(self.embedding_dim)
            self.counts[label] = 1.0
        else: # Blend old centroid with new embedding
            self.centroids[label] = (self.decay_rate * self.centroids[label] + (1 - self.decay_rate) * embedding)
            self.counts[label] += 1.0

    def score(self, embedding):

        # Computing a favorabilty score using similarities dict
        similarities = {}

        for label, centroid in self.centroids.items():

            # Compute euclidean distance
            distance = np.linalg.norm(embedding - centroid)
            # Store as score
            similarities[label] = 1 / (distance + 1e-8) # Lower distance is more similar

        # Compute a weighted average
        total_similarity = sum(similarities.values())
        if total_similarity == 0:

            return 0.5  # Default to neutral if all similarities are 0

        # Weighted average of label values based on similarity
        score = sum(label * similarities[label] for label in similarities) / total_similarity

        return score  # Output is between 0 (bad) and 1 (good)


if __name__ == "__main__":
    map = Mapping()



