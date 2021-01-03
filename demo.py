from images import show_image
from cutobject import CutObject
from segmentation import Segmentation

if __name__ == '__main__':
    MASKRCNN_MODEL = "models/mrcnn/mask_rcnn_coco.h5"

    segmentation = Segmentation()
    matting = CutObject(segmentation, "tmp")

    image_path = "asserts/girl1.jpg"
    foreground_path = matting.cut(image_path)

    show_image(foreground_path)
