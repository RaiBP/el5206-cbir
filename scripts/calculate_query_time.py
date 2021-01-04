import time

from src.similarity_functions import get_similarities
from src.ImageExtractor import ImageExtractor
from src.feature_extractors.HOGExtractor import HOGExtractor
from src.feature_extractors.ResNetExtractor import ResNetExtractor
import os
from os.path import join as osjoin
from tqdm import tqdm
from src.metrics.SimilarityMetrics import EuclideanDistance

extractor = 'hog'
#extractor = 'resnet'

MAIN_DIR = os.path.dirname(os.getcwd())
IMAGES_PATH = osjoin(MAIN_DIR, 'image_database')
FEATURES_PATH = osjoin(MAIN_DIR, 'feature_database')

ie = ImageExtractor(IMAGES_PATH)
fe = HOGExtractor() if extractor == 'hog' else ResNetExtractor()

FEATURES_FILENAME = 'hog_features.txt' if extractor == 'hog' else 'resnet_features.txt'
FEATURES_DATABASE = osjoin(FEATURES_PATH, FEATURES_FILENAME)

distance_metric = EuclideanDistance()


def calculate_time(query_class):
    image_query_path = ie.get_query_image(query_class)

    ti = time.time()
    feature_vector_query = fe.extract_features(image_query_path)
    get_similarities(feature_vector_query, FEATURES_DATABASE, distance_metric)
    tf = time.time()
    return tf - ti


time_results_filename = osjoin(MAIN_DIR, 'scripts', f'{extractor}_time_results.txt')

time_results = open(time_results_filename, 'w+')

for i in tqdm(range(ie.number_of_query_images)):
    query_time = calculate_time(i)
    time_results.write(f'{i},{query_time}\n')
time_results.close()
