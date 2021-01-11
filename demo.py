from cutobject import CutObject
from images import show_image
from segmentation import Segmentation

if __name__ == '__main__':
    MASKRCNN_MODEL = "models/mrcnn/mask_rcnn_coco.h5"

    segmentation = Segmentation()
    matting = CutObject(segmentation)

    image_path = "asserts/girl1.jpg"
    foreground_path = matting.cut(image_path)

    show_image(foreground_path)
