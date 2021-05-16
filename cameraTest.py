#! /usr/bin/env python

import cv2
import numpy as np


def cameraTest():
    cam = cv2.VideoCapture(1)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cam.isOpened():
        print("Cannot open camera!")
        return
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Cannot get frame!")
        cv2.imshow('Camera', frame)
        cv2.waitKey(1)
    cam.release()
    cv2.destroyAllWindows()
    return



if __name__ == '__main__':
    cameraTest()