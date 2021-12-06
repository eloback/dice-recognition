# Authors:
#   - Eduardo Loback Stefani
#   - Enzo Redivo Canella
#   - Khalil Santana

import cv2 as cv
import numpy as np
import os, random    

path = "./dataset/"

#given a image, return all countours.
def getCountours(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    rect, thresh = cv.threshold(gray, 50, 255, cv.THRESH_BINARY)
    #cv.bitwise_not(thresh, thresh)
    #thresh = cv.erode(thresh, (5, 5) iterations=6)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return thresh, contours

#alternative way of getting countours
def getCountoursUsingCanny(image):
    rect, thresh = cv.threshold(image, 125, 255, cv.THRESH_BINARY, image)
    gray = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return edges, contours

#return the original image with the dices segmented and the list of dices.
def segmentAndDrawDices(contours, image):
    rects = []
    for cnt in contours:
        rect = cv.minAreaRect(cnt)
        #filter out rects that don't have enough area to be dices
        if rect[1][0]*rect[1][1] < 500 or rect[1][0]*rect[1][1] > 2000:
            continue
        rects.append(rect)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(image, [box], 0, (0, 0, 255), 2)
    return image, rects

#for 2 random images in dataset, segment the dices and draw them.
for i in range(0, 2):
    file = random.choice(os.listdir(path))
    img = cv.imread(path + file)
    result, contours = getCountours(img)
    segmentedImage, rects = segmentAndDrawDices(contours, img.copy())
    #Show the Original image
    cv.imshow("Original "+file, img)
    cv.moveWindow("Original "+file, 1, img.shape[0]*i+1)
    #show the product of the contours extraction
    cv.imshow(file, result)
    cv.moveWindow(file, img.shape[1], img.shape[0]*i+1)
    #Show the segmented dice
    cv.imshow("contours "+file, segmentedImage)
    cv.moveWindow("contours "+file, img.shape[1]*2+1, img.shape[0]*i+1)
    print("number of dices in "+file + " is " + str(len(rects)))
    
cv.waitKey(0)