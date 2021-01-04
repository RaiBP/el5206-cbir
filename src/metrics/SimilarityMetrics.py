import numpy as np


class CosineDistance:
    @staticmethod
    def calculate_distance(fv1, fv2):
        return np.dot(fv1, fv2) / (np.linalg.norm(fv1) * np.linalg.norm(fv1))


class EuclideanDistance:
    @staticmethod
    def calculate_distance(fv1, fv2):
        return np.linalg.norm(fv1 - fv2)
