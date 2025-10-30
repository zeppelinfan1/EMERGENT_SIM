"""
GENETIC MATERIAL
Provides possible actions for neural network brain to process.
Actions are associated with various properties within the ecosystem.
"""

# IMPORTS
import numpy as np
import random, math
from dataclasses import dataclass, field


# OBJECTS
@dataclass
class Genetics:

    gene_number: int
    gene_length: int
    genes: list = field(init=False)
    mapping: list = field(init=False)
    # Gene metrics
    edge_lengths: list = field(init=False)
    axis_extents: tuple = field(init=False)
    plane_metrics: dict = field(init=False)

    def __post_init__(self):

        # Binary array for each neuron
        self.genes = [self.generate_gene() for _ in range(self.gene_number)]
        # Mapping for each gene
        self.mapping = self.generate_map()
        # Metrics
        self.edge_lengths = self.pairwise_edges(self.mapping)
        self.axis_extents = self.axis_extents_fn(self.mapping)
        self.plane_metrics = self.plane_metrics_from_points(self.mapping)

    @staticmethod
    def bits_to_unit(b):

        n = len(b)
        if n == 0:
            return 0.0
        val = int("".join(map(str, b)), 2)
        max_val = (1 << n) - 1

        return (val / max_val) * 2.0 - 1.0 if max_val > 0 else 0.0

    @staticmethod
    def pairwise_edges(points):

        n = len(points)
        edges = []
        for i in range(n):

            for j in range(i + 1, n):

                xi, yi, zi = points[i]
                xj, yj, zj = points[j]
                d = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)
                edges.append(((i, j), d))

        return edges

    @staticmethod
    def axis_extents_fn(points):

        arr = np.array(points)
        min_vals = arr.min(axis=0)
        max_vals = arr.max(axis=0)

        return tuple(max_vals - min_vals)

    @staticmethod
    def plane_metrics_from_points(points):

        n = len(points)
        if n < 3:
            return {"normal": (0, 0, 0), "area": 0, "centroid": (0, 0, 0), "plane_distance": 0}

        # centroid
        cx = sum(p[0] for p in points) / n
        cy = sum(p[1] for p in points) / n
        cz = sum(p[2] for p in points) / n
        centroid = (cx, cy, cz)

        # build normal by summing cross products of consecutive edges
        nx = ny = nz = 0.0
        area_sum = 0.0
        p0 = points[0]
        for i in range(1, n - 1):
            a = [points[i][j] - p0[j] for j in range(3)]
            b = [points[i + 1][j] - p0[j] for j in range(3)]
            # cross product
            cxp = [a[1] * b[2] - a[2] * b[1],
                   a[2] * b[0] - a[0] * b[2],
                   a[0] * b[1] - a[1] * b[0]]
            nx += cxp[0]
            ny += cxp[1]
            nz += cxp[2]
            area_sum += math.sqrt(cxp[0] ** 2 + cxp[1] ** 2 + cxp[2] ** 2) / 2

        norm = math.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
        if norm != 0:
            nx /= norm
            ny /= norm
            nz /= norm
        plane_distance = abs(nx * p0[0] + ny * p0[1] + nz * p0[2])

        return {
            "normal": (nx, ny, nz),
            "area": area_sum,
            "centroid": centroid,
            "plane_distance": plane_distance
        }

    def generate_gene(self):

        # Random binary list
        return [random.randint(0, 1) for _ in range(self.gene_length)]

    def generate_map(self):

        mapping_list = []
        for gene in self.genes:

            n = len(gene)
            seg = n // 3 or 1
            x_bits = gene[:seg]
            y_bits = gene[seg:2*seg]
            z_bits = gene[2*seg:]
            # Mapping a binary gene to a Hilbert curve coordinate
            x_value = self.bits_to_unit(x_bits)
            y_value = self.bits_to_unit(y_bits)
            z_value = self.bits_to_unit(z_bits)
            mapping_list.append((x_value, y_value, z_value))

        return mapping_list

    def __repr__(self):

        # Formats output
        return f"Genetics(gene_number={self.gene_number}, gene_length={self.gene_length}, genes={self.genes}, mapping={self.mapping})"


# RUN
if __name__ == "__main__":
    g = Genetics(gene_number=3, gene_length=6)
    print("Mapping:", g.mapping)
    print("Edges:", g.edge_lengths)
    print("Axis extents:", g.axis_extents)
    print("Plane metrics:", g.plane_metrics)
