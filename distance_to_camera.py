# import the necessary packages
from imutils import paths
import numpy as np
import imutils
import cv2

def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)
	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key = cv2.contourArea)
	cv2.drawContours(image, [c], 0, (0, 255, 255), 3)
	# compute the bounding box of the of the paper region and return it
	return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth


KNOWN_DISTANCE = 24.0
KNOWN_WIDTH = 11.0

def initialize():
	# initialize the known object width, which in this case, the piece of
	# paper is 12 inches wide

	# load the furst image that contains an object that is KNOWN TO BE 2 feet
	# from our camera, then find the paper marker in the image, and initialize
	# the focal length
	image = cv2.imread("images/2ft.jpg")
	marker = find_marker(image)
	focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
	return focalLength

def find_distance(image, box, focalLength):
	inches = distance_to_camera(KNOWN_WIDTH, focalLength, box[2])
	cv2.putText(image, "%.2fm" % (inches / 39.37),
				(int(box[0] + box[2] + 5), int(box[1] + box[3])), cv2.FastFeatureDetector_TYPE_5_8,
				1.0, (0, 255, 0), 2)
