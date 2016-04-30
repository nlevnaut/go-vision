#!/usr/bin/python3

import numpy as np
import cv2
from cv2 import Canny
from cv2 import bilateralFilter
from cv2 import HoughCircles
from cv2 import GetPerspectiveTransform
from cv2 import WarpPerspective


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

    def orient_board(self):
        '''Morphs the perspective of the board to a square.'''
        # harris corner detector?
        # GetPerspectiveTransform
        # WarpPerspective
        pass

    def board_lines(self):
        pass

    def import_board(self):
        #orient_board()
        #self.white, self.black = find_stones()
        # 
        pass

    # Finds stones in image
    def find_stones(self):
        img2 = np.empty(self.img.shape, 'uint8')
        bilateralFilter(self.img, img2, 9, 75, 75) 
        self.stones = HoughCircles(img2, CV_HOUGH_GRADIENT, 2, img2.shape[1]/21)
        # canny?
        # smooth?
        # houghcircles?
        pass

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
    show_webcam(mirror=True)

if __name__ == '__main__':
    main()
