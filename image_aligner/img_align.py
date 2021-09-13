import cv2
import os


class AlignImage():
    def getSkewAngle(self, cvImage):
        img = cvImage.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)

        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilate = cv2.dilate(thresh, kernel, iterations=5)

        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        largest_contour = contours[0]
        min_area_rect = cv2.minAreaRect(largest_contour)

        angle = min_area_rect[-1]
        if angle < -45:
            angle = 90 + angle
        return -1.0 * angle

    def rotate_image(self, cvImage, angle):
        img = cvImage.copy()
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return img

    def align(self, cvImage):
        angle = self.getSkewAngle(cvImage)
        return self.rotate_image(cvImage, -1.0 * angle)


obj = AlignImage()
path = 'misaligned_images'
i = 0
for file in os.listdir(path):
    if file.endswith(".jpg") or file.endswith(".png"):
        file_path = f"{path}\\{file}"
        aligned_img = obj.align(cv2.imread(file_path))
        cv2.imwrite(f'aligned_images\\output_{i}.jpg', aligned_img)
        i += 1
