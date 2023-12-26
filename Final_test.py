from cvzone import HandTrackingModule
import cv2
import numpy as np
import math 
import time
import keras
import os

m = keras.models.load_model(rf'{os.getcwd()}\best_so_far19th.h5', compile=False)
m.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
print(m.summary())
detector = HandTrackingModule.HandDetector(maxHands=1)
cap = cv2.VideoCapture(0)
offset = 20
imgSize = 200
# imgSize = 64
timer = 0
sl = []
dict1 = {chr(i):0 for i in range(65,65+26)}
def check(x):
    if x == 0:
        return "A"
    elif x == 1:
        return "B"
    return "C"

counter = 0
ct = 0
sy = "!"
hands1 = []
while True:
    data, image = cap.read()
    hands = detector.findHands(image, draw=False)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgBlack = np.ones((imgSize, imgSize, 3), np.uint8)*200
        # imgBlack = image[y-offset:y+imgSize-offset, x-offset:x+-offset+imgSize]
        if x > 0 and y > 0:
            # imgCrop = image[y:y+imgSize, x:x+imgSize]
            imgCrop = image[(y-offset) :(y+h+offset), (x - offset):(x+w+offset)]
        else:
            imgCrop = np.zeros(imgSize)
        imgCropShape = imgCrop.shape
        aspectRatio = h/w
        if aspectRatio > 1:
            k = imgSize/h
            wC = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wC, imgSize))
            imgResizeShape = imgResize.shape
            gap = math.ceil((imgSize-wC)/2)
            imgBlack[:, gap:wC+gap] = imgResize
        else:
            k = imgSize/w
            hC = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hC))
            imgResizeShape = imgResize.shape
            gap = math.ceil((imgSize-hC)/2)
            imgBlack[gap:hC+gap, :] = imgResize
        # print(imgBlack.shape)
        pre = cv2.cvtColor(imgBlack, cv2.COLOR_BGR2GRAY)
        # pre = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
        # print(pre.shape)
        if cv2.waitKey(1) == ord("c"):
            ss = chr(np.argmax(m.predict(pre.reshape(1,imgSize, imgSize), verbose=None))+65)
            # if ss in list(dict1.keys()):
            #     # print(dict1)
            #     dict1[ss]+=1
            # timer+=1
        
            # if(timer==10):
            #     ss = max(zip(dict1.values(), dict1.keys()))[1]
            if ss != sy:
                    print(ss)
                    sy = ss

            # dict1 = dict.fromkeys(dict1, 0)
        cv2.imshow('Handtracker1', imgCrop)
        cv2.imshow('Handtracker2', imgBlack)
    cv2.imshow('Handtracker', image)
    if cv2.waitKey(1) == ord("q"):
        # cv2.imwrite("testfilename.jpg", image) #save image
        break

cap.release()
cv2.destroyAllWindows()
