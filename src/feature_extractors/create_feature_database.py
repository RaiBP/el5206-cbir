import os
import numpy as np

from tqdm import tqdm
from os import listdir
from os.path import isfile, join


def get_database_images(images_path):
    images_filenames = [f for f in listdir(images_path) if isfile(join(images_path, f))]
    database_images = []
    for filename in images_filenames:
        filename_without_extension = filename.split(".")[0]
        image_id = filename_without_extension[-2:]
        if image_id != "00":
            database_images.append(filename)
    return database_images


def create_database(images_path, database_folder, database_filename, feature_extractor):
    if not os.path.exists(database_folder):
        os.makedirs(database_folder)

    features_database = open(database_filename, 'w+')

    database_images = get_database_images(images_path)

    for filename in tqdm(database_images, desc=f'Creating {feature_extractor.name} feature database', leave=True):
        file_path = os.path.join(images_path, filename)
        features = feature_extractor.extract_features(file_path)
        features_database.write(filename.split('.')[0] + ',' + ','.join(str(x) for x in features) + '\n')

    features_database.close()
    print(f'{feature_extractor.name} feature database created in {database_filename}')


def get_feature_vectors_given_image_codes(image_codes, features_database_path):
    database_file = open(features_database_path, 'r')

    image_code_list = []
    fv_list = []

    while True:
        line = database_file.readline()
        if not line:
            break
        data = line.split(",")
        image_code = data[0]

        if image_code in image_codes:
            image_code_list.append(image_code)
            fv_list.append(np.asarray(data[1:]).astype(float))
    database_file.close()
    return image_code_list, fv_list
