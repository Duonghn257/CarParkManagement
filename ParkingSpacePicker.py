import cv2
import pickle

img = cv2.imread('./CarParkProject/carPark2.jpg')
img = cv2.resize(img, None, fx=0.5, fy=0.5)
Img = img.copy()
width, height = 107, 48
draw = False
rect = (0,0,1,1)
rectangle = False
rect_over = False
ix, iy = -1, -1
try:
    with open('./CarParkProject/carPark2', 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []

def mouseClick(events, x, y, flags, params):
    global draw, ix, iy, xx, yy

    if events == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        draw = True
    elif events == cv2.EVENT_RBUTTONDOWN:
        posList.pop()
    elif events == cv2.EVENT_MOUSEMOVE:
        Img = img.copy()
        if draw == True:
            cv2.rectangle(Img, (ix, iy), (x, y), (255, 0, 0), 2)
            cv2.imshow("image", Img)
            cv2.waitKey(1)
    elif events == cv2.EVENT_LBUTTONUP:
        draw = False
        posList.append(((ix, iy), (x, y)))
        cv2.rectangle(Img, (ix, iy), (x, y), (255, 0, 0), 2)
        # cv2.imshow("image", Img)
    with open('./CarParkProject/carPark2', 'wb') as f:
        pickle.dump(posList, f)
cnt = 0
while True:
    cnt+=1

    img = cv2.imread('./CarParkProject/carPark2.jpg')
    # img = cv2.resize(img)
    Img = img.copy()
    for pos in posList:
        cv2.rectangle(Img, pos[0], pos[1], (255, 0, 0), 2)

    if cnt%20 == 0:
        print(posList)
    cv2.imshow("image", Img)
    cv2.setMouseCallback("image", mouseClick)
    cv2.waitKey(10)

