#!/usr/bin/python3

import numpy as np
import cv2
from cv import Canny
from cv import Smooth
from cv import HoughCircles
from cv import GetPerspectiveTransform
from cv import WarpPerspective


class Board(object):
    def __init__(self, img):
        self.img = img
        self.white = []
        self.black = []
        self.import_board()

    def orient_board(self):
        # harris corner detector?
        # GetPerspectiveTransform
        # WarpPerspective

    def import_board(self):
        #orient_board()
        #self.white, self.black = find_stones()
        # 
        # pass

    # Finds stones in image
    def find_stones(self):
        # canny?
        # smooth?
        # houghcircles?
        pass

    # Checks color of a given stone
    def stone_color():
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
