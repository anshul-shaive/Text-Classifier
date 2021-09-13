import math
import os
import cv2
import numpy as np
from deskew import determine_skew


def rotate(image, angle, background):
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)


path = 'misaligned_images'
i = 0
for file in os.listdir(path):
    if file.endswith(".jpg") or file.endswith(".png"):
        file_path = f"{path}\\{file}"
        img = cv2.imread(file_path)
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        skew_angle = determine_skew(grayscale)
        rotated = rotate(img, skew_angle, (0, 0, 0))
        cv2.imwrite(f'deskewed_images\\output_{i}.jpg', rotated)
        i += 1
