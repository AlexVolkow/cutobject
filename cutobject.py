import os

import numpy as np
from pymatting import cutout

from coco.coco import COCO_CLASSES
from images import generate_trimap, expand_mask


class CutObject:
    def __init__(self, segmentation, temp_dir):
        self.segmentation = segmentation
        self.temp_dir = temp_dir

    def cut(self, image_path, crop=None):
        trimap_path = self.temp_dir + "/trimap/"
        foreground_path = self.temp_dir + "/foreground/"

        if not (os.path.exists(trimap_path)):
            os.mkdir(trimap_path)

        if not (os.path.exists(foreground_path)):
            os.mkdir(foreground_path)

        file = os.path.basename(image_path).split(".")[0]
        trimap_image = trimap_path + "trimap_" + file + ".png"

        foreground_name = foreground_path + str(file) + ".png"

        mask = self.predict_mask(image_path, crop)
        mask = np.uint8(mask * 255)
        generate_trimap(mask, trimap_image)

        cutout(image_path, trimap_image, foreground_name)

        return foreground_name

    def predict_mask(self, image_name, crop):
        predict_result = self.segmentation.predict(image_name, crop)
        return self.select_biggest_area_class_mask(predict_result)

    def select_biggest_area_class_mask(self, predict_result):
        index_position = 0
        area_by_class = {}
        max_class_name = ""
        for i in predict_result["class_ids"]:
            class_name = COCO_CLASSES[i]
            mask = predict_result['masks'][:, :, index_position]

            sum_area = area_by_class.get(class_name, 0)
            sum_area += self.area(mask)

            area_by_class[class_name] = sum_area

            if max_class_name == "" or area_by_class[max_class_name] < sum_area:
                max_class_name = class_name

            index_position += 1

        if max_class_name == "":
            raise Exception("Can't detect anything")

        index_position = 0
        final_mask = None
        for i in predict_result["class_ids"]:
            if COCO_CLASSES[i] == max_class_name:
                mask = predict_result['masks'][:, :, index_position]
                mask = expand_mask(mask, 16)

                if final_mask is None:
                    final_mask = np.zeros(mask.shape, np.bool)

                for x in range(0, mask.shape[0]):
                    for y in range(0, mask.shape[1]):
                        if not final_mask[x][y]:
                            final_mask[x][y] = mask[x][y]

            index_position += 1

        return final_mask

    def area(self, mask):
        return mask.sum()
