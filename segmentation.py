import cv2 as cv
import numpy as np
import os, random    

path = "./dataset/"

def segmentDices(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    rect, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
    dices = cv.bitwise_not(thresh)
    return dices

for i in range(0, 2):
    file = random.choice(os.listdir(path))
    img = cv.imread(path + file)
    dices = segmentDices(img)
    #Show the Original image
    cv.imshow("Original"+str(i), img)
    cv.moveWindow("Original"+str(i), 1, img.shape[0]*i+1)
    #Show the segmented dice
    cv.imshow(str(i), dices)
    cv.moveWindow(str(i), img.shape[1], img.shape[0]*i+1)
    
cv.waitKey(0)