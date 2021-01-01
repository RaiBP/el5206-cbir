import cv2
from skimage.feature import hog


class HOGExtractor:
    def __init__(self, desired_height=2048, desired_width=2304, cell_size=256, cells_per_block=3):
        self.desired_height = desired_height
        self.desired_width = desired_width
        self.cell_size = cell_size
        self.cells_per_block = cells_per_block
        self.name = 'HOG'

    def extract_features(self, file_path):
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        image_resized = self.image_resize(image)
        image_padded = self.image_padding(image_resized)
        return hog(image_padded, pixels_per_cell=(self.cell_size, self.cell_size),
                   cells_per_block=(self.cells_per_block, self.cells_per_block), multichannel=True)

    def image_padding(self, image):
        (h, w) = image.shape[:2]

        delta_w = self.desired_width - w
        delta_h = self.desired_height - h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    def image_resize(self, image, inter=cv2.INTER_AREA):
        (h, w) = image.shape[:2]

        if h > w:
            r = self.desired_height / float(h)
            dim = (int(w * r), self.desired_height)

        else:
            r = self.desired_width / float(w)
            dim = (self.desired_width, int(h * r))

        resized = cv2.resize(image, dim, interpolation=inter)

        return resized
