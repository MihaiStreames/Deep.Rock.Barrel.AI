from PIL import Image
import pytesseract
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
img_path = "../Image/1.png"

img = Image.open(img_path)
img = img.convert('RGB')

img_np = np.array(img)

score_area = img_np[150:300, 850:1050]

score_area_gray = cv2.cvtColor(score_area, cv2.COLOR_RGB2GRAY)
_, score_area_thresh = cv2.threshold(score_area_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

score_text = pytesseract.image_to_string(score_area_thresh, config='--psm 6')

try:
    detected_score = int(score_text.strip())
except ValueError:
    detected_score = None

print(detected_score)

cv2.imshow('Score Area', score_area_thresh)
cv2.waitKey(0)