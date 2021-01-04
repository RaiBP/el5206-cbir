import time

import cv2

from src.similarity_functions import get_similarities
from src.ImageExtractor import ImageExtractor
from src.clustering.clustering import get_similarities_with_clustering, create_cluster_database
from src.feature_extractors.HOGExtractor import HOGExtractor
from src.feature_extractors.create_feature_database import create_database
from src.feature_extractors.ResNetExtractor import ResNetExtractor
import os
import sys
from os.path import join as osjoin
import argparse
from src.metrics.SimilarityMetrics import EuclideanDistance, CosineDistance
import matplotlib.pyplot as plt


def check_and_create_successive_folders(main_directory, *argv):
    path = main_directory
    for arg in argv:
        path = osjoin(path, arg)
        if not os.path.exists(path):
            os.mkdir(path)
    return path


def save_results(code_query, similarity_df, images_path, extractor_name, use_clustering):
    fig, axes = plt.subplots(11, 1, figsize=(8, 50))

    for i in range(1, 12):
        if i == 1:
            image = cv2.imread(osjoin(images_path, code_query + ".jpg"), cv2.IMREAD_COLOR)

            axes[i - 1].title.set_text(f'Imagen de consulta: {code_query}.jpg')
        else:
            image_index = i - 2
            code = similarity_df.iloc[image_index, :]['image_code']
            image = cv2.imread(osjoin(images_path, code + ".jpg"), cv2.IMREAD_COLOR)
            axes[i - 1].title.set_text(f'Coincidencia {i - 1}: {code}.jpg')
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[i - 1].imshow(rgb_img)
        axes[i - 1].axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0.1, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    save_folder = check_and_create_successive_folders(MAIN_DIR, 'query_results', code_query)

    clustering_suffix = '_clustering' if use_clustering else ''
    filename = osjoin(save_folder, f'{code_query}_10_best_matches_{extractor_name}{clustering_suffix}.png')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    print(f'Results saved in {filename}')


def check_valid_class(class_id):
    ivalue = int(class_id)
    if ivalue < 0 or ivalue > 499:
        raise argparse.ArgumentTypeError("%s is an invalid class" % class_id)
    return ivalue


def check_query_image_exists(image_path, image_filename):
    return os.path.exists(osjoin(image_path, image_filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get 10 similar images.')
    parser.add_argument('query_class', metavar='N', type=check_valid_class,
                        help='class of image query (integer between 0 and 499)')
    parser.add_argument('extractor', choices=['hog', 'resnet'],
                        help="feature extractor, either "
                             "'hog' or 'resnet'")
    parser.add_argument('-db', '--generate_database', help="calculate feature vectors and generate new features "
                                                           "database",
                        action="store_true")
    parser.add_argument('-di', '--download_images', help="forces a re-download of the image dataset",
                        action="store_true")
    parser.add_argument('-cl', '--clustering', help="use k-means clustering", action="store_true")
    parser.add_argument('-cldb', '--clustering_new_database',
                        help="use k-means clustering and force creation of new clustering database",
                        action="store_true")
    parser.add_argument('--distance', choices=['euclidean', 'cosine'], default='euclidean',
                        help="similarity metric, either "
                             "'euclidean' or 'cosine'")

    args = parser.parse_args()

    MAIN_DIR = os.getcwd()
    IMAGES_PATH = osjoin(MAIN_DIR, 'image_database')

    download_images = not os.path.exists(IMAGES_PATH) or args.download_images

    FEATURES_PATH = osjoin(MAIN_DIR, 'feature_database')

    ie = ImageExtractor(IMAGES_PATH, download_images=download_images)

    if ie.total_number_of_images == 0:
        sys.exit(f'Image dataset directory {IMAGES_PATH} is empty\nPlease try again with added argument -di or '
                 f'--download_images to force a re-download of the image dataset')

    class_id_format = "{:03d}".format(args.query_class)
    image_code_query = "1" + class_id_format + "00"

    if not check_query_image_exists(IMAGES_PATH, f'{image_code_query}.jpg'):
        sys.exit(f"Query image with filename {image_code_query}.jpg couldn't be found in path {IMAGES_PATH}\nPlease "
                 f"try again with added argument -di or --download_images to force a re-download of the image dataset")

    fe = HOGExtractor() if args.extractor == 'hog' else ResNetExtractor()

    image_query_path = ie.get_query_image(args.query_class)

    FEATURES_FILENAME = 'hog_features.txt' if args.extractor == 'hog' else 'resnet_features.txt'
    FEATURES_DATABASE = osjoin(FEATURES_PATH, FEATURES_FILENAME)

    DATABASE_EXISTS = os.path.exists(FEATURES_DATABASE)

    if not DATABASE_EXISTS or args.generate_database:
        if DATABASE_EXISTS:
            os.remove(FEATURES_DATABASE)
        create_database(IMAGES_PATH, FEATURES_PATH, FEATURES_DATABASE, fe)

    distance_metric = EuclideanDistance() if args.extractor == 'euclidean' else CosineDistance()

    if args.clustering or args.clustering_new_database:
        CLUSTERING_DATABASE = osjoin(MAIN_DIR, 'clustering_database')
        CLUSTERING_DATABASE_MAIN = osjoin(CLUSTERING_DATABASE, f'{args.extractor}_clustering_database.txt')
        CLUSTERING_DATABASE_CENTROIDS = osjoin(CLUSTERING_DATABASE,
                                               f'{args.extractor}_clustering_database_centroids.txt')

        if not os.path.exists(CLUSTERING_DATABASE_MAIN) or not os.path.exists(CLUSTERING_DATABASE_CENTROIDS) \
                or args.clustering_new_database:
            print(f'Creating {fe.name} clustering database')
            create_cluster_database(FEATURES_DATABASE, CLUSTERING_DATABASE_MAIN, n_clusters=16)
            print(f'{fe.name} feature database created in {CLUSTERING_DATABASE}')
        ti = time.time()
        feature_vector_query = fe.extract_features(image_query_path)
        similarities = get_similarities_with_clustering(feature_vector_query, CLUSTERING_DATABASE_CENTROIDS,
                                                        CLUSTERING_DATABASE_MAIN, FEATURES_DATABASE, distance_metric)
    else:
        ti = time.time()
        feature_vector_query = fe.extract_features(image_query_path)
        similarities = get_similarities(feature_vector_query, FEATURES_DATABASE, distance_metric)
    tf = time.time()

    print(f'Query completed in {tf - ti} secs')

    for index in range(10):
        row = similarities.iloc[index, :]
        distance = row['distance']
        image_code = row['image_code']
        rank = row['rank']
        print(f'Image {image_code}, rank {rank}, distance {distance}')

    save_results(image_code_query, similarities, IMAGES_PATH, args.extractor, args.clustering)
