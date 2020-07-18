import cv2
import numpy as np
from misc import onClick
from matplotlib import pyplot as plt
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Read in image and get shape
img = cv2.imread("./img/level_box.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (300, 300))
img = cv2.GaussianBlur(img, (11, 11), 0)
img = cv2.medianBlur(img, 9)


output = pytesseract.image_to_string(img, config='--psm 8 digits')
print(f"Output: {output}")

cv2.imshow("Round", img)

cv2.waitKey()
cv2.destroyAllWindows()