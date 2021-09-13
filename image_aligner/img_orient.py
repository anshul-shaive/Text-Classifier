import os
import cv2
import numpy as np


def rotate(img, theta):
    rows, cols = img.shape[0], img.shape[1]
    image_center = (cols / 2, rows / 2)
    mat = cv2.getRotationMatrix2D(image_center, theta, 1)
    abs_cos = abs(mat[0, 0])
    abs_sin = abs(mat[0, 1])
    bound_w = int(rows * abs_sin + cols * abs_cos)
    bound_h = int(rows * abs_cos + cols * abs_sin)
    mat[0, 2] += bound_w / 2 - image_center[0]
    mat[1, 2] += bound_h / 2 - image_center[1]
    rotated = cv2.warpAffine(img, mat, (bound_w, bound_h), borderValue=(255, 255, 255))
    return rotated


def slope(x1, y1, x2, y2):
    if x1 == x2:
        return 0
    slope = (y2 - y1) / (x2 - x1)
    theta = np.rad2deg(np.arctan(slope))
    return theta


def main(filePath):
    img = cv2.imread(filePath)
    text_img = img.copy()

    small = cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(bw.shape, dtype=np.uint8)
    theta_s = 0

    ct = 0
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y + h, x:x + w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y + h, x:x + w])) / (w * h)
        if r > 0.45 and w > 8 and h > 8:
            rect = cv2.minAreaRect(contours[idx])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(text_img, [box], 0, (0, 0, 255), 2)

            theta = slope(box[0][0], box[0][1], box[1][0], box[1][1])
            theta_s += theta
            ct += 1

    orientation = theta_s / ct
    rotated_img = rotate(img, orientation)
    return rotated_img


path = 'misaligned_images'
i = 0
for file in os.listdir(path):
    if file.endswith(".jpg") or file.endswith(".png"):
        file_path = f"{path}\\{file}"
        image = main(file_path)
        cv2.imwrite(f'oriented_images\\output_{i}.jpg', image)
        i += 1
