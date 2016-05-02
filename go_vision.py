#!/usr/bin/python3

import sys
import argparse
import numpy as np
from PIL import Image
import cv2
from cv2 import getPerspectiveTransform
from cv2 import warpPerspective
from cv2 import cornerHarris
from cv2 import imread, imshow, waitKey, destroyAllWindows



class Board(object):
    '''
    Go board class.
    Takes numpy img array as input.
    '''
    def __init__(self, img):
        self.img = img
        self.stones = []
        self.white = []
        self.black = []
        self.import_board()

    def read(self, filename):
        '''
        Function to load an image of a Go board.
        '''
        self.img = imread(filename)

    def align_board(self):
        '''
        Morphs the perspective of the board to a square.
        Result stored in self.aligned.
        '''
        board_contour = None
        # Smooth and histogram equalize before thresholding
        self.smooth = cv2.bilateralFilter(self.gray, 3, 20, 20)
        #self.equalized = cv2.equalizeHist(self.gray)

        # Threshold and then detect contours, to get board
        #ret, self.thresh = cv2.threshold(self.equalized, 114, 255, 0)
        self.thresh = cv2.adaptiveThreshold(self.smooth, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 1)
        #kernel = np.ones((5,5),np.uint8)
        #self.ethresh = cv2.erode(self.thresh,kernel,iterations=1)
        im2, contours, hier = cv2.findContours(self.thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Make sure the contour is big enough, but ignore the border of the image
            if area > 50000 and area < 230000:
                hull = cv2.convexHull(cnt)
                hull = cv2.approxPolyDP(hull, 0.1*cv2.arcLength(hull, True), True)
                if len(hull == 4):
                    board_contour = hull

        points = board_contour.reshape(4,2)
        rect = np.zeros((4,2), dtype='float32')
        sumpts = points.sum(axis=1)
        rect[0] = points[np.argmin(sumpts)]
        rect[2] = points[np.argmax(sumpts)]
        diffpts = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diffpts)]
        rect[3] = points[np.argmax(diffpts)]
        
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

        # ...and now for the height of our new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

        # take the maximum of the width and height values to reach
        # our final dimensions
        max_width = max(int(widthA), int(widthB))
        max_height = max(int(heightA), int(heightB))

        dst = np.array(
            [[0,0],
             [max_width-1,0],
             [max_width-1,max_height-1],
             [0,max_height-1]],
            dtype = 'float32')

        M = cv2.getPerspectiveTransform(rect, dst)
        self.aligned = cv2.warpPerspective(self.img, M, (max_width, max_height))
        self.aligned = cv2.resize(self.aligned,(400,400))
        self.alignedgray = cv2.cvtColor(self.aligned, cv2.COLOR_BGR2GRAY)
        return


    def board_lines(self):
        pass

    def find_white(self):
        pass

    # Finds stones in image
    def find_stones(self):
        #stoneimg = np.empty(self.aligned.shape, 'uint8')
        #bilateralFilter(self.img, img2, 9, 75, 75) 

        # Smooth, aligned gray image
        sag = cv2.bilateralFilter(self.alignedgray, 3, 20, 20)

        kernel = np.ones((3,3),np.uint8)
        
        #mask1 = cv2.inRange(sag, 0, 85)
        mask1 = cv2.inRange(sag, 100, 255)
        mask2 = cv2.inRange(sag, 140, 255)

        self.blackrange = cv2.bitwise_not(sag)
        #self.blackrange = cv2.bitwise_and(self.blackrange, self.blackrange, mask=mask1)
        #self.blackrange = cv2.bitwise_and(self.alignedgray, sag, mask=mask1)
        #self.blackrange = cv2.erode(self.blackrange,kernel,iterations=2)
        #self.blackrange = cv2.dilate(self.blackrange,kernel,iterations=2)
        #self.blackrange = cv2.equalizeHist(self.blackrange)
        #self.blackrange = cv2.bitwise_not(self.blackrange, self.blackrange)
        ret, self.blackrange = cv2.threshold(self.blackrange, 210, 255, 0)


        self.whiterange = cv2.bitwise_and(sag, sag, mask=mask2)
        ret, self.whiterange = cv2.threshold(self.whiterange, 140, 255, 0)

        self.black = cv2.HoughCircles(self.blackrange,cv2.HOUGH_GRADIENT,1,16,param1=30,param2=5,minRadius=7,maxRadius=11) 
        self.white = cv2.HoughCircles(self.whiterange,cv2.HOUGH_GRADIENT,1,16,param1=30,param2=5,minRadius=7,maxRadius=11)

        '''
        self.stones = cv2.HoughCircles(self.alignedgray, cv2.HOUGH_GRADIENT, 2, 
                                        minDist=int(self.aligned.shape[1]/38),
                                        #minRadius=int(self.aligned.shape[1]/42), 
                                        #maxRadius=int(self.aligned.shape[1]/3))
        '''
        return

    def import_board(self):
        self.img = cv2.resize(self.img, (500,500))
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.align_board()
        self.find_stones()
        for i in self.white[0,:]:
            cv2.circle(self.aligned, (i[0],i[1]),i[2],(0,255,0),1)
            cv2.circle(self.aligned, (i[0],i[1]),2,(0,0,255),2)

        for i in self.black[0,:]:
            cv2.circle(self.aligned, (i[0],i[1]),i[2],(255,0,0),1)
            cv2.circle(self.aligned, (i[0],i[1]),2,(0,255,0),2)

    # Checks color of a given stone
    def stone_color():
        for stone in self.stones:
            pass

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('webcam', img)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()

def main():
    try:
        filename = sys.argv[1]
    except:
        print("Usage: %s img" % sys.argv[0])
        exit()
    try:
        img = imread(filename)
    except:
        print("Something bad happened")
        exit()
    if img == None:
        print("You didn't actually pass an image, dumbass")

    board = Board(img)
    cv2.imshow('resized image', board.img)
    cv2.imshow('thresholded image', board.thresh)
    cv2.imshow('aligned image', board.aligned)
    cv2.imshow('black image', board.blackrange)
    cv2.imshow('white image', board.whiterange)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
