import cv2 as cv
import numpy as np
import os, random    

path = "./dataset/"

def segmentDiceUsingCanny(image):
    rect, thresh = cv.threshold(image, 125, 255, cv.THRESH_BINARY, image)
    gray = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    return edges

def segmentDice(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    rect, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
    return thresh

def extractDices(segmentedImage, image):
    copy = image.copy()
    contours, hierarchy = cv.findContours(segmentedImage, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        rect = cv.minAreaRect(cnt)
        #filter out rects that don't have enough area to be dices
        if rect[1][0]*rect[1][1] < 500:
            continue
        rects.append(rect)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(copy, [box], 0, (0, 0, 255), 2)
    return copy, rects

for i in range(0, 2):
    file = random.choice(os.listdir(path))
    img = cv.imread(path + file)
    segmentedImage = segmentDice(img)
    result, rects = extractDices(segmentedImage, img)
    print("number of dices in "+str(i) + " is " + str(len(rects)))
    #print countours on a blank image
    cv.imshow("contours"+str(i), result)
    cv.moveWindow("contours"+str(i), img.shape[1]*2+1, img.shape[0]*i+1)
    #Show the Original image
    cv.imshow("Original"+str(i), img)
    cv.moveWindow("Original"+str(i), 1, img.shape[0]*i+1)
    #Show the segmented dice
    cv.imshow(str(i), segmentedImage)
    cv.moveWindow(str(i), img.shape[1], img.shape[0]*i+1)
    
cv.waitKey(0)