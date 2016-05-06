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
from math import floor

class Board(object):
    '''
    Go board class.
    Takes numpy img array as input.
    '''
    def __init__(self, img):
        self.img = img
        self.stones = []
        self.white = np.array((2,2))
        self.black = np.array((2,2))
        self.positions = []
        self.moves = 0
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
        # Smooth before thresholding
        self.smooth = cv2.bilateralFilter(self.gray, 5, 20, 20)

        # Threshold and then detect contours, to get board
        self.thresh = cv2.adaptiveThreshold(self.smooth, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 1)
        im2, contours, hier = cv2.findContours(self.thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Make sure the contour is big enough, but ignore the border of the image
            if area > 50000 and area < 230000:
                hull = cv2.convexHull(cnt)
                hull = cv2.approxPolyDP(hull, 0.1*cv2.arcLength(hull, True), True)
                if len(hull == 4):
                    self.board_contour = hull

        # This perspective warp is based off the pokedex tutorial at pyimagesearch.com.
        points = self.board_contour.reshape(4,2)
        rect = np.zeros((4,2), dtype='float32')
        # Get top-left and bottom-right points of rectangle.
        sumpts = points.sum(axis=1)
        rect[0] = points[np.argmin(sumpts)]
        rect[2] = points[np.argmax(sumpts)]
        # Get top-right and bottom-left points of rectangle.
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
        max_width = max(int(widthA), int(widthB)) + 25
        max_height = max(int(heightA), int(heightB)) + 25

        dest = np.array(
            [[0,0],
             [max_width-1,0],
             [max_width-1,max_height-1],
             [0,max_height-1]],
            dtype = 'float32')

        M = cv2.getPerspectiveTransform(rect, dest)
        self.aligned = cv2.warpPerspective(self.img, M, (max_width, max_height))
        self.aligned = cv2.resize(self.aligned,(400,400))
        self.alignedgray = cv2.cvtColor(self.aligned, cv2.COLOR_BGR2GRAY)
        return

    def find_stones(self):
        '''
        Finds stones in thresholded black and white images.
        Results stored in self.black and self.white.
        '''
        # Smooth, aligned gray image
        sag = cv2.bilateralFilter(self.alignedgray, 3, 20, 20)

        kernel = np.ones((3,3),np.uint8)
        
        #mask1 = cv2.inRange(sag, 0, 85)
        mask1 = cv2.inRange(sag, 100, 255)
        mask2 = cv2.inRange(sag, 140, 255)

        self.blackrange = cv2.bitwise_not(sag)
        ret, self.blackrange = cv2.threshold(self.blackrange, 210, 255, 0)
        self.blackrange = cv2.erode(self.blackrange,kernel,iterations=3)
        self.blackrange = cv2.dilate(self.blackrange,kernel,iterations=3)

        #self.whiterange = cv2.bitwise_and(sag, sag, mask=mask2)
        ret, self.whiterange = cv2.threshold(sag, 135, 255, 0)
        self.whiterange = cv2.erode(self.whiterange,kernel,iterations=2)
        self.whiterange = cv2.dilate(self.whiterange,kernel,iterations=2)

        self.black = cv2.HoughCircles(self.blackrange,cv2.HOUGH_GRADIENT,1,16,param1=30,param2=7,minRadius=4,maxRadius=13)[0] 
        self.white = cv2.HoughCircles(self.whiterange,cv2.HOUGH_GRADIENT,1,16,param1=30,param2=7,minRadius=4,maxRadius=13)[0]

    def circle_pos(self, circle, color):
        '''
        Returns the coordinates of a circle, using SGF notation.
        Currently only works for 19x19 boards.
        '''
        # margins
        marg = 23
        # image size
        imgsize = 400
        horiz = ['a','b','c','d','e','f','g','h','i','j','k',
                'l','m','n','o','p','q','r','s']
        verti = horiz
        
        cpos = []
        # Find the index for both horizontal and vertical components of the circle.
        i1 = floor(((circle[0])/(imgsize-marg))*19+0.5)-1
        i2 = floor(((circle[1])/(imgsize-marg))*19+0.5)-1
        if i1 < 0:
            i1 = 0
        elif i1 > 18:
            i1 = 18
        if i2 < 0:
            i2 = 0
        elif i2 > 18:
            i2 = 18
        cpos.append(color)
        cpos.append(horiz[i1])
        cpos.append(verti[i2])
        return tuple(cpos)

    # Align the board and find stones. Save SGF moves to self.boardstring.
    def import_board(self):
        '''
        Imports a go board, stored a self.img.
        Calls align_board() and find_stones(), then generates an image with
        circles drawn where they were detected in the original.
        '''
        self.img = cv2.resize(self.img, (500,500))
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.align_board()
        self.find_stones()
        self.circleimg = self.aligned.copy()

        # Draw circles on the original image.
        for i in self.white:
            self.positions.append(self.circle_pos(i, 'W'))
            cv2.circle(self.circleimg, (i[0],i[1]),i[2],(0,255,0),1)
            cv2.circle(self.circleimg, (i[0],i[1]),2,(0,0,255),2)

        for i in self.black:
            self.positions.append(self.circle_pos(i, 'B'))
            cv2.circle(self.circleimg, (i[0],i[1]),i[2],(255,0,0),1)
            cv2.circle(self.circleimg, (i[0],i[1]),2,(0,255,0),2)

        # Save SGF string
        self.boardstring = ''
        count = 1
        for p in self.positions:
            self.moves += 1
            self.boardstring += ';' + p[0] + '[' + p[1] + p[2] + ']'
            if count % 11 == 0:
                self.boardstring += '\n'
            count += 1

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
        print("You didn't actually pass an image")

    board = Board(img)
    print('(;GM[1]FF[4]\nSZ[19]\nGN[go-vision 1.0]\nKM[0.0]HA[0]RU[Japanese]AP[GNU Go:3.8]')
    print(board.boardstring + ')')
    cv2.imshow('resized image', board.img)
    cv2.waitKey(0)
    cv2.imshow('thresholded image', board.thresh)
    cv2.waitKey(0)
    cv2.drawContours(board.img, [board.board_contour], -1, (0, 255, 0), 3)
    cv2.imshow("Game Boy Screen", board.img)
    cv2.waitKey(0)
    cv2.imshow('aligned image', board.aligned)
    cv2.waitKey(0)
    cv2.imshow('black image', board.blackrange)
    cv2.waitKey(0)
    cv2.imshow('white image', board.whiterange)
    cv2.waitKey(0)
    cv2.imshow('circle image', board.circleimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
