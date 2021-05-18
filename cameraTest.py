#! /usr/bin/env python

import cv2
import numpy as np

FRAME_WIDTH = 1800
FRAME_HEIGHT = 1000

EYE_THRES_WIDTH_L = 0.15
EYE_THRES_WIDTH_R = 0.05
EYE_THRES_HEIGHT = 0.15
STARTTHRES = 40


def cameraTest():
	cam = cv2.VideoCapture(0, cv2.CAP_V4L2)
	cv2.namedWindow('Camera')
	cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
	cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
	if not cam.isOpened():
		print("Cannot open camera!")
		return
	detector_params = cv2.SimpleBlobDetector_Params()
	detector_params.filterByArea = True
	detector_params.maxArea = 1500
	detector = cv2.SimpleBlobDetector_create(detector_params)
	startThres = STARTTHRES
	CALIBRATED = False
	two_eyes_detected_count = 0
	while True:
		ret, frame = cam.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
		# for (x,y,w,h) in faces:	# Detect one face for now
		if len(faces) == 1:
			(x,y,w,h) = faces[0]
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
			face = frame[y:y+h, x:x+w]ector_params)

			face_h, face_w, face_channel = face.shape
			eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
			eyes_detected_num = 0
			for eye_detected in eyes:
				if eye_detected is not None:
					(eye_x, eye_y, eye_w, eye_h) = eye_detected
					if eye_y >= int(y+h/2):
						continue
					eye = frame[eye_y+int(EYE_THRES_HEIGHT*eye_h):eye_y+int((1-EYE_THRES_HEIGHT)*eye_h)
								, eye_x+int(EYE_THRES_WIDTH_L*eye_w):eye_x+int((1-EYE_THRES_WIDTH_R)*eye_w)]
					
					gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
					thres_eye = cv2.threshold(gray_eye, startThres, 255, cv2.THRESH_BINARY)[1]
					thres_eye = cv2.erode(thres_eye, None, iterations=2)
					thres_eye = cv2.dilate(thres_eye, None, iterations=4)
					thres_eye = cv2.GaussianBlur(thres_eye, (3,3), 0)
					keypoints = detector.detect(thres_eye)
					cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
					if keypoints:
						eyes_detected_num += 1
			if keypoints and CALIBRATED:
				total_x = total_y = 0
				for keypoint in keypoints:
					total_x += keypoint.pt[0]
					total_y += keypoint.pt[1]
				cm = (int(total_x/len(keypoints)), int(total_y/len(keypoints)))
				cm_frame = (int((1-cm[0]/eye_w)*FRAME_WIDTH), int(cm[1]/eye_h*FRAME_HEIGHT))
				print(eye_x, eye_w, eye_y, eye_h, cm, cm_frame)
				frame = cv2.circle(frame, cm_frame, 30, (0, 0, 255), 10)
			if eyes_detected_num < 2 and not CALIBRATED:
				startThres += 5
			elif eyes_detected_num == 2 and not CALIBRATED:
				two_eyes_detected_count += 1
				if two_eyes_detected_count >= 3:
					startThres += 5
					CALIBRATED = True
			cv2.putText(frame, str(startThres),(10,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
			cv2.imshow('Camera', frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		if not ret:
			print("Cannot get frame!")
		
	cam.release()
	cv2.destroyAllWindows()
	return

def nothing(x):
	pass


if __name__ == '__main__':
	cameraTest()