import subprocess
import tarfile
from os import listdir
from os.path import isfile, join

import cv2
import shutil
import os
from os.path import join as osjoin

MAIN_DIR = os.getcwd()


class ImageExtractor:
    def __init__(self, images_path, download_images=False):
        self.images_path = images_path

        if not os.path.exists(self.images_path):
            os.mkdir(self.images_path)

        if download_images:
            self.generate_image_database()

        self.number_of_query_images, self.number_of_database_images = self.calculate_numbers_of_images()
        self.total_number_of_images = self.number_of_query_images + self.number_of_database_images

    def download_from_url(self, url):
        process = subprocess.Popen(['wget', url],
                                   stdout=subprocess.PIPE,
                                   universal_newlines=True, cwd=self.images_path)
        while True:
            line = process.stdout.readline()
            if not line:
                break
            print(line)

    def extract_file(self, file_path):
        with tarfile.open(file_path, 'r:gz') as f:
            f.extractall(path=self.images_path)

    @staticmethod
    def move_files(source_dir, target_dir):
        file_names = os.listdir(source_dir)

        for file_name in file_names:
            shutil.move(osjoin(source_dir, file_name), target_dir)

    def download_and_extract(self, url, filename, description):
        print(f'Downloading {description}...')
        self.download_from_url(url)
        filepath = osjoin(self.images_path, filename)
        print(f'Extracting of {description}...')
        self.extract_file(filepath)
        print(f'Extraction of {description} completed')
        os.remove(filepath)

    def generate_image_database(self):
        url1 = 'ftp://ftp.inrialpes.fr/pub/lear/douze/data/jpg1.tar.gz'
        self.download_and_extract(url1, 'jpg1.tar.gz', 'first tar.gz file')
        url2 = 'ftp://ftp.inrialpes.fr/pub/lear/douze/data/jpg2.tar.gz'
        self.download_and_extract(url2, 'jpg2.tar.gz', 'second tar.gz file')


        self.move_files(osjoin(self.images_path, 'jpg'), self.images_path)
        os.rmdir(osjoin(self.images_path, 'jpg'))

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
