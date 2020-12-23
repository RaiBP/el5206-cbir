import numpy as np


class EuclideanDistance:
    @staticmethod
    def calculate_distance(feature_vector1, feature_vector2):
        return np.linalg.norm(feature_vector1 - feature_vector2)
