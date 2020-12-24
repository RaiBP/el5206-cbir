import os

from src.ImageExtractor import ImageExtractor
from src.feature_extractors.HOGExtractor import HOGExtractor
from metrics import get_normalized_rank
from src.metrics.EuclideanDistance import EuclideanDistance

from tqdm import tqdm

MAIN_DIR = os.path.dirname(os.getcwd())

database_name = 'hog_features.txt'
#database_name = 'resnet_features.txt'
database_folder = os.path.join(MAIN_DIR, 'feature_database', database_name)
images_folder = os.path.join(MAIN_DIR, 'image_database')

ie = ImageExtractor(images_folder)

rank_norm_list = []

distance_metric = EuclideanDistance()
total_number_of_images = ie.number_of_query_images + ie.number_of_database_images
feature_extractor = HOGExtractor()
#feature_extractor = ResNetExtractor()

#rank_results_filename = 'resnet_rank_results.txt'
rank_results_filename = 'hog_rank_results.txt'

rank_results = open(os.path.join(MAIN_DIR, rank_results_filename), 'w+')

for i in tqdm(range(500)):
    image_query = ie.get_query_image(i)
    fv_query = feature_extractor.extract_features(image_query)
    norm_rank = get_normalized_rank(fv_query, i, database_folder, distance_metric, total_number_of_images)
    rank_results.write(f'{i},{norm_rank}\n')
rank_results.close()
