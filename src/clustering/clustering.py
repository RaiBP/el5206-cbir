import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans

from src.similarity_functions import get_similarities_for_given_feature_vectors, get_normalized_rank_given_similarities
from src.feature_extractors.create_feature_database import get_feature_vectors_given_image_codes

MAIN_PATH = os.path.dirname(os.path.dirname(os.getcwd()))


def get_feature_vector_matrix(feature_vector_database_path):
    database_file = open(feature_vector_database_path, 'r')

    image_code_list = []
    fv_list = []

    while True:
        line = database_file.readline()

        if not line:
            break

        data = line.split(",")
        image_code = data[0]
        feature_vector = np.asarray(data[1:]).astype(float)

        fv_list.append(feature_vector)
        image_code_list.append(image_code)
    return np.asarray(fv_list), image_code_list


def create_cluster_database(feature_vector_database_path, clustering_database_filename, n_clusters):
    fv_matrix, code_list = get_feature_vector_matrix(feature_vector_database_path)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(fv_matrix)
    labels = kmeans.labels_
    kcenters = kmeans.cluster_centers_

    database_folder = os.path.join(MAIN_PATH, 'clustering_database')
    if not os.path.exists(database_folder):
        os.makedirs(database_folder)

    clustering_database = open(os.path.join(database_folder, clustering_database_filename), 'w+')

    for index, code in enumerate(code_list):
        clustering_database.write(f'{code},{labels[index]}\n')

    clustering_database.close()

    centroids_database = open(os.path.join(database_folder, clustering_database_filename.split(".")[0] +
                                           "_centroids.txt"), 'w+')

    for cluster_index in range(n_clusters):
        centroids_database.write(f'{cluster_index},' + ','.join(str(x) for x in kcenters[cluster_index, :]) + '\n')
    centroids_database.close()


def get_distances_to_centroids(fv_query, centroid_database_path, similarity_metric):
    database_file = open(centroid_database_path, 'r')

    distances_list = []
    cluster_indices = []

    while True:
        line = database_file.readline()
        if not line:
            break
        data = line.split(",")
        cluster_index = int(data[0])
        centroid_vector = np.asarray(data[1:]).astype(float)
        distance = similarity_metric.calculate_distance(centroid_vector, fv_query)
        distances_list.append(distance)
        cluster_indices.append(cluster_index)
    database_file.close()
    return pd.Series(distances_list, index=cluster_indices).sort_values()


def get_closest_centroid(fv_query, centroid_database_path, similarity_metric):
    return get_distances_to_centroids(fv_query, centroid_database_path, similarity_metric).index[0]


def get_image_codes_for_given_centroid(query_cluster_index, cluster_database_path):
    database_file = open(cluster_database_path, 'r')

    image_code_list = []

    while True:
        line = database_file.readline()
        if not line:
            break
        data = line.split(",")
        image_code = data[0]
        centroid_index = int(data[1])

        if centroid_index == query_cluster_index:
            image_code_list.append(image_code)
    database_file.close()
    return image_code_list


def get_feature_vectors_for_given_centroid(query_centroid_index, features_database_path, cluster_database_path):
    image_codes = get_image_codes_for_given_centroid(query_centroid_index, cluster_database_path)
    img_in_cluster_codes, fv_in_cluster = get_feature_vectors_given_image_codes(image_codes, features_database_path)
    return img_in_cluster_codes, fv_in_cluster


def get_similarities_for_given_centroid(query_centroid_index, feature_vector_query, features_database_path,
                                        cluster_database_path, similarity_metric):
    img_codes, fv_in_cluster = get_feature_vectors_for_given_centroid(query_centroid_index, features_database_path,
                                                                      cluster_database_path)
    similarities = get_similarities_for_given_feature_vectors(feature_vector_query, fv_in_cluster, similarity_metric)

    return img_codes, similarities


def get_similarities_with_clustering(fv_query, centroid_database_path, cluster_database_path, features_database_path,
                                     similarity_metric):
    best_centroid = get_closest_centroid(fv_query, centroid_database_path, similarity_metric)
    img_codes, similarities = get_similarities_for_given_centroid(best_centroid, fv_query, features_database_path,
                                                                  cluster_database_path, similarity_metric)
    similarity_df = pd.DataFrame({'distance': similarities,
                                  'image_code': img_codes}).sort_values(by='distance').reset_index()
    del similarity_df['index']
    similarity_df['rank'] = similarity_df.index + 1
    return similarity_df


def get_normalized_rank_with_clustering(fv_query, class_query, features_database_path, cluster_database_path,
                                        centroid_database_path, similarity_metric, number_of_images):
    centroid_distances = get_distances_to_centroids(fv_query, centroid_database_path, similarity_metric)

    similarity_df_list = []

    for idx in range(len(centroid_distances)):
        current_centroid = centroid_distances.index[idx]
        img_code, similarity = get_similarities_for_given_centroid(current_centroid, fv_query, features_database_path,
                                                                   cluster_database_path, similarity_metric)

        similarity_df = pd.DataFrame({'distance': similarity,
                                      'image_code': img_code}).sort_values(by='distance').reset_index()
        similarity_df_list.append(similarity_df)

    total_similarities_df = pd.concat(similarity_df_list).reset_index()
    del total_similarities_df['index']
    total_similarities_df['rank'] = total_similarities_df.index + 1

    return get_normalized_rank_given_similarities(total_similarities_df, class_query, number_of_images)
