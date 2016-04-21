# -*- coding: utf-8 -*-

from __init__ import _EYECASCADE, _FACECASCADE

import cv2
import numpy


class WebCamTracker:
	
	"""
	"""
	
	def __init__(self, camnr=0):
		
		"""
		"""
		
		# Initialise the webcam.
		self._vidcap = cv2.VideoCapture(camnr)
		self._connected = True
		
		# Initialise the cascading algorithms for face and eye detection.
		# NOTE: Credit for and an explanation of the following classifiers
		# can be found on OpenCV's documentation website. Please see:
		# http://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html
		self._face_cascade = cv2.CascadeClassifier(_FACECASCADE)
		self._eye_cascade = cv2.CascadeClassifier(_EYECASCADE)
	
	
	def close(self):
		
		"""
		"""
		
		# Release the video capture from the current webcam.
		self._vidcap.release()
		self._connected = False

	
	def is_connected(self):
		
		"""
		"""
		
		return self._connected

	
	def _get_frame(self):
		
		"""Reads the next frame from the active video capture.
		
		Returns
		
		success, frame	-	success is a Boolean that indicates whether
						a frame could be obtained.
						frame is a numpy.ndarray with unsigned,
						8-bit integers that reflect the greyscale
						values of the image.
		"""
		
		# Take a photo with the webcam.
		# (ret is the return value: True if everything went ok, False if
		# there was a problem. frame is the image taken from the webcam as
		# a NumPy ndarray, where the image is coded as BGR
		ret, frame = self._vidcap.read()
		
		# Convert to grey scale.
		if ret:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		return ret, frame


	def _crop_face(self, frame, minsize=(30, 30)):
		
		"""Attempts to find faces in the image, and crops the largets.
		
		Arguments

		frame			-	A numpy.ndarray with unsigned, 8-bit
						integers that reflect the greyscale values
						of the image.
		
		Keyword Arguments
		
		minsize		-	A (w,h) tuple that indicates what size a
						face should minimally be. Default = (30,30)
		
		Returns
		
		success, crop	-	success is a Boolean that indicates whether
						a face could be detected in the frame.
						crop is a numpy.ndarray with unsigned,
						8-bit integers that reflect the greyscale
						values of the largest rect in the image
						where a face could be detected.
		"""
		
		# Find all potential faces in the frame.
		faces = self._face_cascade.detectMultiScale(
			frame,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=minsize,
			flags = cv2.cv.CV_HAAR_SCALE_IMAGE
			)
		
		# Find out what the largest face is (this is assumed to be the
		# actual face when multiple faces are present).
		if len(faces) == 0:
			success = False
			x, y, w, h = 0, 0, frame.shape[1], frame.shape[0]
		else:
			success = True
			lh = 0
			for i in range(len(faces)):
				if faces[i][3] > lh:
					x, y, w, h = faces[i]
		
		# Return the cropped frame.
		return success, frame[y:y+h, x:x+w]
	
	
	def _crop_eyes(self, facecrop, Lexpect=(0.7,0.4), Rexpect=(0.3,0.4), \
		maxdist=0.2, maxsize=0.3):
		
		"""Attempts to find eyes in an image of a face, and crops the left
		and the right eye separately. When more than two potential eyes
		are detected, the eyes that are closest to the expected positions
		of the eyes will be selected.
		
		Arguments

		facecrop		-	A numpy.ndarray with unsigned, 8-bit
						integers that reflect the greyscale values
						of a face.
		
		Keyword Arguments
		
		Lexpect		-	A (x,y) tuple that indicates where the left
						eye is expected to be. Note that the
						coordinates are in relative space, where
						(0,0) is the top-left of the image, (0,1)
						is the bottom-left, and (1,1) is the
						bottom-right. Also note that the left eye is
						likely to be on the right side of the image,
						and the right eye is likely to be in the
						left part of the image. Default = (0.7,0.4)
		
		Rexpect		-	A (x,y) tuple that indicates where the right
						eye is expected to be. Note that the
						coordinates are in relative space, where
						(0,0) is the top-left of the image, (0,1)
						is the bottom-left, and (1,1) is the
						bottom-right. Also note that the left eye is
						likely to be on the right side of the image,
						and the right eye is likely to be in the
						left part of the image. Default = (0.3,0.4)
		
		maxdist		-	A float that indicates what the maximal
						allowable distance is between the expected
						eye position, and the position of detected
						potential eye. The maximal distance is
						defined as a proportion of the image height.
						It can also be set to None. Default = (0.2)
		
		maxsize		-	A float that indicates what the maximal
						allowable width is of the detected eyes. The
						maximal size is defined as a proportion of
						the image width. It can also be set to None.
						Default = (0.3)
		
		Returns
		
		success, [left, right]	-	success is a Boolean that indicates 
							whether the eyes could be detected.
							left and right are both a numpy.ndarray
							with unsigned, 8-bit integers that
							reflect the greyscale values of what
							are assumed to be the left and the
							right eye.
		"""
		
		# DETECT THE EYES
		eyes = self._eye_cascade.detectMultiScale(facecrop)
		
		# Remove all the potential eye rects that are too large.
		if maxsize != None:
			eyes = eyes[eyes[:,3] < maxsize * facecrop.shape[0]]
		
		# Calculate the distances between each potential eye and the
		# expected locations. (NOTE: These are not the actual distances,
		# but the squared distances. They need to be compared to the
		# squared maximum distance.)
		cx = (eyes[:,0] + eyes[:,2] / 2) / float(facecrop.shape[1])
		cy = (eyes[:,1] + eyes[:,3] / 2) / float(facecrop.shape[0])
		dl = (cx - Lexpect[0])**2 + (cy - Lexpect[1])**2
		dr = (cx - Rexpect[0])**2 + (cy - Rexpect[1])**2
		# Exclude all potential eyes that are too far from expected eye
		# locations.
		if maxdist != None:
			good = numpy.min([dl, dr], axis=0) < maxdist**2
			eyes = eyes[good]
			dl = dl[good]
			dr = dr[good]
		
		# COUNT THE EYES
		# If no eye was detected, there is no eye index.
		if len(eyes) == 0:
			li = None
			ri = None
		
		# If only one eye is detected, its index is 0.
		elif len(eyes) == 1:
			# Check whether the distance to the left eye is closer. If it
			# is, than only the left eye was recorded. If not, the right
			# eye was recorded.
			if dl[0] < dr[0]:
				li = 0
				ri = None
			else:
				li = None
				ri = 0
		
		# If two or more eyes were detected, choose the potential rects
		# that were closest to the expected eye positions.
		else:
			li = numpy.argmin(dl)
			ri = numpy.argmin(dr)

		# RETURN CROPPED EYES
		# If no eye was detected, return no success.
		success = True
		if (li == None) & (ri == None):
			success = False
		# If the left eye was detected, crop it from the face.
		if li == None:
			left = None
		else:
			x, y, w, h = eyes[li]
			left = facecrop[y:y+h, x:x+w]
		# If the right eye was detected, crop it from the face.
		if ri == None:
			right = None
		else:
			x, y, w, h = eyes[ri]
			right = facecrop[y:y+h, x:x+w]

		return success, [left, right]


# # # # #
# DEBUG #
if __name__ == u'__main__':
	import os
	import time
	from matplotlib import pyplot
#	filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), u'test.jpeg')
	tracker = WebCamTracker()
#	img = cv2.imread(filepath)
#	frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	success = False
	while not success:
		t0 = time.time()
		success, frame = tracker._get_frame()
	if success:
		success, facecrop = tracker._crop_face(frame)
		success, eyes = tracker._crop_eyes(facecrop)
	t1 = time.time()
	tracker.close()
	print("Elapsed time: %.3f ms" % (1000*(t1-t0)))
	pyplot.figure(); pyplot.imshow(facecrop, cmap='gray')
	pyplot.figure(); pyplot.imshow(eyes[0], cmap='gray')
	pyplot.figure(); pyplot.imshow(eyes[1], cmap='gray')
# # # # #