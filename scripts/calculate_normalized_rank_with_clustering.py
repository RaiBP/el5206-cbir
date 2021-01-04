import os

from src.ImageExtractor import ImageExtractor
from src.clustering.clustering import get_normalized_rank_with_clustering
from src.feature_extractors.HOGExtractor import HOGExtractor
from src.feature_extractors.ResNetExtractor import ResNetExtractor
from src.metrics.SimilarityMetrics import EuclideanDistance
from os.path import join as osjoin

from tqdm import tqdm

MAIN_DIR = os.path.dirname(os.getcwd())

# extractor = 'hog'
extractor = 'resnet'

CLUSTERING_DATABASE = osjoin(MAIN_DIR, 'clustering_database')
CLUSTERING_DATABASE_MAIN = osjoin(CLUSTERING_DATABASE, f'{extractor}_clustering_database.txt')
CLUSTERING_DATABASE_CENTROIDS = osjoin(CLUSTERING_DATABASE, f'{extractor}_clustering_database_centroids.txt')

FEATURES_PATH = osjoin(MAIN_DIR, 'feature_database')
FEATURES_FILENAME = f'{extractor}_features.txt'
FEATURES_DATABASE = osjoin(FEATURES_PATH, FEATURES_FILENAME)

IMAGES_PATH = osjoin(MAIN_DIR, 'image_database')

ie = ImageExtractor(IMAGES_PATH)

rank_norm_list = []

distance_metric = EuclideanDistance()
total_number_of_images = ie.number_of_query_images + ie.number_of_database_images
feature_extractor = HOGExtractor() if extractor == 'hog' else ResNetExtractor()

rank_results_filename = osjoin(MAIN_DIR, 'scripts', f'{extractor}_rank_clustering_results.txt')

rank_results = open(rank_results_filename, 'w+')

for i in tqdm(range(ie.number_of_query_images)):
    image_query = ie.get_query_image(i)
    fv_query = feature_extractor.extract_features(image_query)
    norm_rank = get_normalized_rank_with_clustering(fv_query, i, FEATURES_DATABASE, CLUSTERING_DATABASE_MAIN,
                                                    CLUSTERING_DATABASE_CENTROIDS, distance_metric,
                                                    total_number_of_images)
    rank_results.write(f'{i},{norm_rank}\n')
rank_results.close()
