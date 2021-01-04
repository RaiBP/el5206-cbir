from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input

from keras.preprocessing import image as image_preprocessing

import numpy as np


class ResNetExtractor:
    def __init__(self):
        self.model = ResNet50(weights='imagenet', include_top=False)
        self.name = 'ResNet'

    def extract_features(self, file_path):
        img_preprocessed = image_preprocessing.load_img(file_path, target_size=(224, 224))
        x = image_preprocessing.img_to_array(img_preprocessed)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features_vector = self.model.predict(x)
        features_averaged = np.apply_over_axes(np.mean, features_vector.squeeze(), [0, 1]).flatten().astype(float)
        return features_averaged
