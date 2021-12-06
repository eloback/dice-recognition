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
    return contours, hierarchy

#alternative way of getting countours
def getCountoursUsingCanny(image):
    rect, thresh = cv.threshold(image, 125, 255, cv.THRESH_BINARY, image)
    gray = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

#return the original image with the dices segmented and the list of dices.
def segmentDices(contours):
    dices = []
    for cnt in contours:
        rect = cv.minAreaRect(cnt)
        #filter out rects that don't have enough area to be dices
        area = rect[1][0]*rect[1][1]
        if area < 500 or area > 2000:
            continue
        dices.append(rect)

    return dices

#return the image of a die, given the contour, including its rotation.
def extractDieImage(rect, image):
    rows, cols = image.shape[:2]
    rotation = cv.getRotationMatrix2D(rect[0], rect[2], 1)
    rotated = cv.warpAffine(image, rotation, (cols, rows))
    w, h = rect[1]
    cropped = cv.getRectSubPix(rotated, (int(w), int(h)), rect[0])
    return cropped

#recents the image of a die, returning its contours and hierarchy.
def getContoursDie(dieImage):
    gray = cv.cvtColor(dieImage, cv.COLOR_BGR2GRAY)
    rect, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def getDieNumber(contours, hierarchy):
    # for each contour, try to fit a rect
    dotRects = []
    for cnt in contours:
        # get the min area rect
        rect = cv.minAreaRect(cnt)
        # filter out rects that don't have enough area to be dices
        area = rect[1][0]*rect[1][1]
        if area < 8 or area > 150:
            continue
        dotRects.append(rect)
    return dotRects



#for 2 random images in dataset, segment the dices and draw them.
for i in range(0, 2):
    file = random.choice(os.listdir(path))
    img = cv.imread(path + file)
    contours, hierarchy = getCountours(img)
    dices = segmentDices(contours)
    diceImages = []
    for die in dices:
        diceImages.append(extractDieImage(die, img))
        dieImg = extractDieImage(die, img)

    #Show the Original image
    cv.imshow("Original "+file, img)
    cv.moveWindow("Original "+file, 1, img.shape[0]*i+1)
    #show the product of the contours extraction
    #Show the segmented image
    cv.imshow("Segmented "+file, cv.drawContours(img, contours, -1, (0, 255, 0), 3))
    cv.moveWindow("Segmented "+file, img.shape[1], img.shape[0]*i+1)
    print("number of dices in "+file + " is " + str(len(dices)))
    #show dices
    for j in range(0, len(diceImages)):
        cv.imshow("Dice "+str(j)+" "+file, diceImages[j])
        kernel = np.ones((3,3), np.uint8)
        dieContours, dieHierarchy = getContoursDie(cv.dilate(diceImages[j], kernel, iterations=1))
        dieNumber = getDieNumber(dieContours, dieHierarchy)
        print("Dice "+str(j)+" has the number "+str(len(dieNumber)))
        cv.imshow("DiceContours "+str(j)+" "+file, cv.drawContours(diceImages[j], dieContours, -1, (0, 255, 0), 3))
        #write dice number
        cv.moveWindow("Dice "+str(j)+" "+file, img.shape[1]*2+1+(33*j), img.shape[0]*i+1)
        cv.moveWindow("DiceContours "+str(j)+" "+file, img.shape[1]*2+1+(33*j), img.shape[0]*i+64)


    
cv.waitKey(0)
