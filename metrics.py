import numpy as np
import pandas as pd


def get_similarities(feature_vector_query, database_path, similarity_metric):
    database_file = open(database_path, 'r')

    image_code_list = []
    distance_list = []

    while True:
        line = database_file.readline()

        if not line:
            break

        data = line.split(",")
        image_code = data[0]
        feature_vector = np.asarray(data[1:]).astype(float)
        distance = similarity_metric.calculate_distance(feature_vector, feature_vector_query)

        image_code_list.append(image_code)
        distance_list.append(distance)
    database_file.close()

    similarity_df = pd.DataFrame({'distance': distance_list,
                                  'image_code': image_code_list}).sort_values(by='distance').reset_index()
    del similarity_df['index']
    similarity_df['rank'] = similarity_df.index + 1
    return similarity_df


def get_similarities_for_given_feature_vectors(feature_vector_query, feature_vector_list, similarity_metric):
    similarity_list = []
    for feature_vector in feature_vector_list:
        similarity_list.append(similarity_metric.calculate_distance(feature_vector, feature_vector_query))
    return similarity_list


def get_same_class_similarities(similarity_df, image_class):
    image_class_format = "{:03d}".format(image_class)
    return similarity_df[similarity_df['image_code'].str.startswith('1' + image_class_format, na=False)]


def get_rank_from_similarities_of_same_class(similarity_df_class):
    return similarity_df_class['rank'].mean()


def get_normalized_rank_given_similarities(similarity_df, image_class, number_of_images):
    similarity_df_class = get_same_class_similarities(similarity_df, image_class)
    rank = get_rank_from_similarities_of_same_class(similarity_df_class)
    return (1 / number_of_images) * (rank - (len(similarity_df_class) + 1) / 2)


def get_normalized_rank(fv_query, class_query, database_path, similarity_metric, number_of_images):
    similarity_df = get_similarities(fv_query, database_path, similarity_metric)
    return get_normalized_rank_given_similarities(similarity_df, class_query, number_of_images)
