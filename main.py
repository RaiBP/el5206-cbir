from src.feature_extractors.create_feature_database import create_database
from src.feature_extractors.ResNetExtractor import ResNetExtractor

feature_extractor = ResNetExtractor()

create_database('image_database', 'resnet_features.txt', feature_extractor)