import cv2

path = "./time.png"
img = cv2.imread(path)
img = cv2.resize(img, dsize=(40, 40))

cv2.imshow("img", img)
cv2.imwrite("./time2.png", img)