import os
import numpy as np

from tqdm import tqdm
from os import listdir
from os.path import isfile, join

MAIN_DIR = os.getcwd()


def get_database_images(images_path):
    images_filenames = [f for f in listdir(images_path) if isfile(join(images_path, f))]
    database_images = []
    for filename in images_filenames:
        filename_without_extension = filename.split(".")[0]
        image_id = filename_without_extension[-2:]
        if image_id != "00":
            database_images.append(filename)
    return database_images


def create_database(images_path, database_filename, feature_extractor):
    database_folder = os.path.join(MAIN_DIR, 'feature_database')
    if not os.path.exists(database_folder):
        os.makedirs(database_folder)

    features_database = open(os.path.join(database_folder, database_filename), 'w+')

    database_images = get_database_images(images_path)

    for filename in tqdm(database_images):
        file_path = os.path.join(images_path, filename)
        features = feature_extractor.extract_features(file_path)
        features_averaged = np.apply_over_axes(np.mean, features, [0, 1]).flatten().astype(float)
        features_database.write(filename.split('.')[0] + ',' + ','.join(str(x) for x in features_averaged) + '\n')

    features_database.close()
