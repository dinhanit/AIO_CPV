from inference import Detect
import cv2

img = cv2.imread(rf"sample\2.jpg")
result = Detect(img)
print(result)
