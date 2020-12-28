import logging
from os.path import exists

import cv2

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

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir="logs/", config=config)

        # Load weights trained on MS-COCO
        self.model.load_weights(model_path, by_name=True)

    def predict(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        # Run detection

        results = self.model.detect([image], verbose=0)

        # Visualize results
        r = results[0]

        return r
