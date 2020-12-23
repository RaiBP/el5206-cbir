from src.feature_extractors import HOGExtractor
from src.feature_extractors import ResNetExtractor


def extract_features(image, extractor_type):
    if extractor_type == 'hog':
        extractor = HOGExtractor()
    elif extractor_type == 'resnet':
        extractor = ResNetExtractor()
    else:
        return None
    return extractor.extract(image)


def similarity_metric(feature_vector1, feature_vector2):
    pass
