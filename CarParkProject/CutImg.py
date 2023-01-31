import cv2



cap = cv2.VideoCapture("E:/TaiLieuHocTap/doantotnghiep/CarParkProject/carPark1.mp4")
while True:
    success, img = cap.read()
    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        cv2.imwrite("./CarPark.jpg", img)

