from images import show_image
from matting import PyMatting
from segmentation import Segmentation

if __name__ == '__main__':
    MASKRCNN_MODEL = "models/mrcnn/mask_rcnn_coco.h5"

    segmentation = Segmentation()
    matting = PyMatting(segmentation)

    image_path = "asserts/girl1.jpg"
    temp_path = "tmp"
    foreground_path = matting.matting(image_path, temp_path)

    show_image(foreground_path)
