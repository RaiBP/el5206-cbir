import os
from os import listdir
from os.path import isfile, join

import cv2


class ImageExtractor:
    def __init__(self, images_path):
        self.images_path = images_path
        self.number_of_query_images, self.number_of_database_images = self.calculate_numbers_of_images()

    def calculate_numbers_of_images(self):
        count_query = 0
        count_database = 0
        images_filenames = [f for f in listdir(self.images_path) if isfile(join(self.images_path, f))]
        for filename in images_filenames:
            filename_without_extension = filename.split(".")[0]
            image_id = filename_without_extension[-2:]
            if image_id == "00":
                count_query += 1
            else:
                count_database += 1
        return count_query, count_database

    def get_query_image(self, class_id):
        class_id_format = "{:03d}".format(class_id)
        filename = "1" + class_id_format + "00.jpg"
        file_path = os.path.join(self.images_path, filename)
        return file_path

    def get_database_image(self, class_id, image_id):
        class_id_format = "{:03d}".format(class_id)
        image_id_format = "{:02d}".format(image_id)
        filename = "1" + class_id_format + image_id_format + ".image_database"
        file_path = os.path.join(self.images_path, filename)
        return cv2.imread(file_path, cv2.IMREAD_COLOR)
