import logging
from os.path import exists

import cv2
import numpy as np

import mrcnn.model as modellib
from coco.coco import CocoConfig
from models import load_mrccn_weight
from mrcnn import utils


class Segmentation:
    def __init__(self):
        model_path = load_mrccn_weight()
        if not exists(model_path):
            logging.info("Load trained weight...")
            utils.download_trained_weights(model_path)
        else:
            logging.info("Load weight from {}".format(model_path))

        config = CocoConfig()
        config.display()

        self.model = modellib.MaskRCNN(mode="inference", model_dir="logs/", config=config)

        self.model.load_weights(model_path, by_name=True)

    def predict(self, image_path, crop=None):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        cropped = not (crop is None)

        if cropped:
            x1, y1, x2, y2 = crop
            image = image[x1:x2, y1:y2]

        results = self.model.detect([image], verbose=0)

        r = results[0]

        if cropped:
            masks = np.zeros((image.shape[0], image.shape[1], len(r["class_ids"])), np.bool)

            index_position = 0
            for _ in r["class_ids"]:
                mask = r['masks'][:, :, index_position]
                for x in range(0, mask.shape[0]):
                    for y in range(0, mask.shape[1]):
                        masks[x + x1][y + y1][index_position] = mask[x][y]
                index_position += 1

            r['masks'] = masks

        return r
