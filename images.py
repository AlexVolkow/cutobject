import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.draw
from skimage.measure import find_contours


def generate_trimap(segment_mask, trimap_name):
    iterations = 8
    alpha = segment_mask

    k_dilated_size = 3
    kernel_dilated = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dilated_size, k_dilated_size))
    dilated = cv2.dilate(alpha, kernel_dilated, iterations=iterations)

    k_erode_size = 5
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_erode_size, k_erode_size))
    eroded = cv2.erode(alpha, kernel_erode, iterations=iterations)

    trimap = np.zeros(alpha.shape, dtype=np.uint8)
    trimap.fill(128)

    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0

    cv2.imwrite(trimap_name, trimap)


def expand_mask(mask, pixels):
    padded_mask = np.zeros(
        (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5)
    for verts in contours:
        verts = np.fliplr(verts) - 1
        rr, cc = skimage.draw.polygon(verts[:, 1], verts[:, 0])
        for j in range(len(rr)):
            x = rr[j]
            y = cc[j]
            mask[x][y] = 1
            for k in range(1, pixels):
                if x - k > 0:
                    mask[x - k][y] = 1
                if x + k < mask.shape[0] and y + k < mask.shape[1]:
                    mask[x + k][y + k] = 1
                if x + k < mask.shape[0]:
                    mask[x + k][y] = 1
                if y + k < mask.shape[1]:
                    mask[x][y + k] = 1
                if x - k > 0 and y - k > 0:
                    mask[x - k][y - k] = 1
                if y - k > 0:
                    mask[x][y - k] = 1
    return mask


def show_image(image_path):
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.show()
