import os
import shutil
import urllib


def load_mrccn_weight():
    weight_dir = "models/mrcnn"
    if not (os.path.exists(weight_dir)):
        os.makedirs(weight_dir)

    weight_path = os.path.join(weight_dir, "mask_rcnn_coco.h5")
    if os.path.exists(weight_path):
        return weight_path

    print("Downloading weight to " + weight_path + " ...")

    weight_url = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"
    with urllib.request.urlopen(weight_url) as resp, open(weight_path, 'wb') as out:
        shutil.copyfileobj(resp, out)
    print("... done downloading.")

    return weight_path
