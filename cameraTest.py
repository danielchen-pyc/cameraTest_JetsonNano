#! /usr/bin/env python

import cv2
import numpy as np

EYE_THRES_WIDTH = 0.1
EYE_THRES_HEIGHT = 0.1


def cameraTest():
	cam = cv2.VideoCapture(0)
	cv2.namedWindow('Camera')
	cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
	cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
	if not cam.isOpened():
		print("Cannot open camera!")
		return
	detector_params = cv2.SimpleBlobDetector_Params()
	detector_params.filterByArea = True
	detector_params.maxArea = 1500
	detector = cv2.SimpleBlobDetector_create(detector_params)
	cv2.createTrackbar('threshold', 'Camera', 0, 255, nothing)
	while True:
		ret, frame = cam.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
		for (x,y,w,h) in faces:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
			gray_face = gray[y:y+h, x:x+w] # cut the gray face frame out
			face = frame[y:y+h, x:x+w] # cut the face frame out
			
			face_h, face_w, face_channel = face.shape
			eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
			max_x = 0
			right_eye = None
			keypoints_list = []
			for eye_detected in eyes:
				if eye_detected is not None:
					(eye_x, eye_y, eye_w, eye_h) = eye_detected
					if eye_y >= int(y+h/2):
						continue
					# if eye_x > max_x:
					#     max_x = eye_x
					#     right_eye = (eye_x, eye_y, eye_w, eye_h)
					# cv2.rectangle(frame, (eye_x, eye_y), (eye_x+eye_w, eye_y+eye_h), (250,250,0), 2)
					eye = frame[eye_y+int(EYE_THRES_HEIGHT*eye_h):eye_y+int((1-EYE_THRES_HEIGHT)*eye_h)
								, eye_x+int(EYE_THRES_WIDTH*eye_w):eye_x+int((1-EYE_THRES_WIDTH)*eye_w)]
					# w, h, c = eye.shape
					# eye = eye[int(EYE_THRES_HEIGHT*h):int((1-EYE_THRES_HEIGHT)*h)
					#             , int(EYE_THRES_WIDTH*w):int((1-EYE_THRES_WIDTH)*w)]
					gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
					threshold = cv2.getTrackbarPos('threshold', 'Camera')
					thres_eye = cv2.threshold(gray_eye, threshold, 255, cv2.THRESH_BINARY)[1]
					thres_eye = cv2.erode(thres_eye, None, iterations=2)
					thres_eye = cv2.dilate(thres_eye, None, iterations=4)
					thres_eye = cv2.GaussianBlur(thres_eye, (5,5), 0)
					keypoints = detector.detect(thres_eye)
					keypoints_list.append(keypoints)
					cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
			cv2.imshow('Camera', frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			# Get one eye
			# if right_eye is not None:
			#     (eye_x, eye_y, eye_w, eye_h) = right_eye
				
			#     cv2.imshow('Eye', eye)
			#     cv2.waitKey(1)
		if not ret:
			print("Cannot get frame!")
		
	cam.release()
	cv2.destroyAllWindows()
	return

def nothing(x):
	pass


if __name__ == '__main__':
	cameraTest()