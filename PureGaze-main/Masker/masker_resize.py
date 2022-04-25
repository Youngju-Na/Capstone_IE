import cv2

img = cv2.imread('./eth-masker.jpg')

img = cv2.resize(img, (448,448), interpolation=cv2.INTER_CUBIC)

cv2.imwrite('./mpii-masker_448.jpg', img)
print(img.shape)