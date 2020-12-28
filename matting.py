import os

import numpy as np
from pymatting import cutout

from coco.coco import COCO_CLASSES
from images import generate_trimap, expand_mask


class PyMatting:
    def __init__(self, segmentation):
        self.segmentation = segmentation

    def matting(self, image_path, temp_dir):
        trimap_path = temp_dir + "/trimap/"
        foreground_path = temp_dir + "/foreground/"

        if not(os.path.exists(trimap_path)):
            os.mkdir(trimap_path)

        if not (os.path.exists(foreground_path)):
            os.mkdir(foreground_path)

        file = os.path.basename(image_path).split(".")[0]
        trimap_image = trimap_path + "trimap_" + file + ".png"

        foreground_name = foreground_path + str(file) + ".png"

        self.predict_mask(image_path, trimap_image, wishlist = ['person'])
        cutout(image_path, trimap_image, foreground_name)

        return foreground_name

    def predict_mask(self, image_name, trimap_name, wishlist=None):
        predict_result = self.segmentation.predict(image_name)

        if wishlist is None:
            wishlist = COCO_CLASSES

        index_position = 0
        #todo combine mask
        #todo move trimap generation
        for i in predict_result['class_ids']:

            if COCO_CLASSES[i] in wishlist:
                mask = predict_result['masks']
                _m = mask[:, :, index_position]
                _m = expand_mask(_m, 5)
                mask = np.uint8(_m * 255)
                generate_trimap(mask, trimap_name)

            index_position += 1
