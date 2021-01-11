import numpy as np
from PIL import Image
from pymatting import estimate_alpha_cf, estimate_foreground_ml, stack_images

from coco.coco import COCO_CLASSES
from images import generate_trimap, expand_mask


class CutObject:
    def __init__(self, segmentation):
        self.segmentation = segmentation

    def cut(self, image, crop=None):
        mask = self.predict_mask(image, crop)
        mask = np.uint8(mask * 255)

        image = image / 255.0

        trimap = generate_trimap(mask)
        trimap = trimap / 255.0

        alpha = estimate_alpha_cf(image, trimap)
        foreground = estimate_foreground_ml(image, alpha)

        result = stack_images(foreground, alpha)
        result = np.clip(result * 255, 0, 255).astype(np.uint8)

        return Image.fromarray(result)

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

    @staticmethod
    def area(mask):
        return mask.sum()
