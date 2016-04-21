# -*- coding: utf-8 -*-

from __init__ import _message, _DEBUGDIR, _EYECASCADE, _FACECASCADE

import cv2
import numpy
from scipy import ndimage


class EyeTracker:
	
	"""Generic class that is intended to act as a parent class for other
	implementations. The general concept of processing images to find eyes
	in them is the same whether the input is a webcam stream, a video, or a
	series of images.
	"""
	
	def __init__(self, pupthreshold=50, glintthreshold=200, debug=False, \
		**kwargs):
		
		"""Initialises an EyeTracker class.
		
		Keyword Arguments
		
		pupthreshold	-	An integer that indicates what the highest
						luminance value is that is still considered
						to be part of the pupil. This value needs to
						be between 0 and 255. Default = 50.
		
		glintthreshold	-	An integer that indicates what the lowest
						luminance value is that is still considered
						to be part of the glint. This value needs to
						be between 0 and 255. Default = 200.
		
		debug			-	A Boolean that indicates whether DEBUG mode
						is active. In DEBUG mode, images from all
						stages of the pupil-extraction process will
						be stored.
		"""
		
		# Set some settings for pupil detection.
		self._pupt = pupthreshold
		self._glit = glintthreshold

		# Initialise the cascading algorithms for face and eye detection.
		# NOTE: Credit for and an explanation of the following classifiers
		# can be found on OpenCV's documentation website. Please see:
		# http://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html
		self._face_cascade = cv2.CascadeClassifier(_FACECASCADE)
		self._eye_cascade = cv2.CascadeClassifier(_EYECASCADE)
		
		# In DEBUG mode, create a Matplotlib figure.
		self._debug = debug
		if self._debug:
			import os
			from matplotlib import patches, pyplot
			global os, patches, pyplot
			self._fig, self._ax = pyplot.subplots(nrows=2, ncols=3)
		
		# Run the custom initialisation procedure.
		self._connected = False
		self.connect(**kwargs)
	
	
	def close(self):
		
		"""Use this function to implement the ending of a specific type
		of eye tracking.
		"""
		
		# Only intialise the EyeTracker if it isn't initialised yet.
		if self._connected:
			self._connected = False
		
		# CUSTOM IMPLEMENTATION HERE
	
	
	def connect(self, **kwargs):
		
		"""Use this function to implement the initialisation of a specific
		type of eye tracking.
		"""
		
		# Only intialise the EyeTracker if it isn't initialised yet.
		if not self._connected:
			self._connected = True
		
			# CUSTOM IMPLEMENTATION HERE

	
	def is_connected(self):
		
		"""Tells you whether the EyeTracker is still connected with its
		source of images.
		
		Returns
		
		connected		-	A Boolean that indicates whether the tracker
						is still connected.
		"""
		
		return self._connected

	
	def _get_frame(self, **kwargs):
		
		"""Use this function to implement how the EyeTracker should obtain
		a frame. A frame is supposed to be a two-dimensional image, usually
		an image from a webcam or a video that was converted to greyscale.
		Note that a frame should be a numpy.ndarray with shape=(w,h), and
		dtype='uint8'. In addition to the frame, a success Boolean should
		be returned by this function. It tells the functions that call
		_get_frame whether a new frame was available. (See below what the
		returned values should be exactly.)
		
		Returns
		
		success, frame	-	success is a Boolean that indicates whether
						a frame could be obtained.
						frame is a numpy.ndarray with unsigned,
						8-bit integers that reflect the greyscale
						values of the image. If no frame could be
						obtained, None will be returned.
		"""
		
		# Obtain a frame.
		_message(u'message', u'generic.EyeTracker.get_frame', \
			u"Implement your own _get_frame functionality")
		
		return False, None


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

		# In DEBUG mode, draw the original frame and the detected faces.
		if self._debug:
			# Display the frame in the top-left pane.
			self._ax[0][0].set_title("potential faces")
			self._ax[0][0].imshow(frame, cmap='gray')
			for i in range(len(faces)):
				r = faces[i]
				self._ax[0][0].add_patch(patches.Rectangle( \
					(r[0],r[1]), r[2], r[3], fill=False, linewidth=1))
			self._ax[0][0].add_patch(patches.Rectangle( \
				(r[0],r[1]), r[2], r[3], fill=False, linewidth=3))
		
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
			y += h/4
			h /= 2
			left = facecrop[y:y+h, x:x+w]
		# If the right eye was detected, crop it from the face.
		if ri == None:
			right = None
		else:
			x, y, w, h = eyes[ri]
			y += h/4
			h /= 2
			right = facecrop[y:y+h, x:x+w]

		# In DEBUG mode, draw the original frame and the detected eyes.
		if self._debug:
			# Display the face in the bottom-left pane.
			self._ax[1][0].imshow(facecrop, cmap='gray')
			self._ax[1][0].set_title("potential eyes")
			for i in range(len(eyes)):
				r = eyes[i]
				if i == li or i == ri:
					self._ax[1][0].add_patch(patches.Rectangle( \
						(r[0],r[1]), r[2], r[3], fill=False, \
						linewidth=3))
				else:
					self._ax[1][0].add_patch(patches.Rectangle( \
						(r[0],r[1]), r[2], r[3], fill=False, \
						linewidth=1))

		return success, [left, right]
	
	
	def _find_pupils(self, left, right, glint=True, mode='diameter'):
		
		"""Finds the pupils and the glints in image of the left and the
		right eye, and returns the relevant parameters. Parameters that
		cannot be found will be None.
		
		Arguments
		
		left			-	A numpy.ndarray with unsigned, 8-bit
						integers that reflect the greyscale values
						of what is assumed to be the left eye.
		
		right			-	A numpy.ndarray with unsigned, 8-bit
						integers that reflect the greyscale values
						of what is assumed to be the right eye.
		
		Keyword Arguments
		
		glint			-	A Boolean that indicates whether the glint
						(the corneal reflection) should also be
						detected. Default = True.
		
		mode			-	A string that indicates how the pupil size
						should be reported.
						'diameter' reports the width of the rect in
						which the thresholded pupil fits
		
		Returns
		[L, R]		-	A NumPy array that contains the following:
		L = [px, py, ps, gx, gy]
		R = [px, py, ps, gx, gy]
					-	px is an integer indicating the pupil's
						likely horizontal position within the image.
						py is an integer indicating the pupil's
						likely vertical position within the image.
						ps is an integer indicating the pupil's
						size in pixels. The size reflects the width
						of the rect in which the pupil falls when
						mode=='diameter', or the total amount of
						thresholded pixels when mode'surface'.
						gx is an integer indicating the pupil's
						likely horizontal position within the image.
						gy is an integer indicating the pupil's
						likely vertical position within the image.
						All of these values can be None if they
						could not be obtained.
		"""
		
		# Create an empty list to hold values for each of the eyes.
		B = numpy.zeros((2, 5))
		i = 0
		for eyeimg in [left, right]:
			if eyeimg == None:
				B[i,:] *= numpy.NaN
			else:
				B[i,:] = self._process_eye_image(eyeimg, glint=True, \
					mode='diameter', debugindex=i)
			i += 1
		
		# In DEBUG mode, save and reset the image.
		if self._debug:
			# Remove all axes' ticklabels
			for i in range(len(self._ax)):
				for j in range(len(self._ax[i])):
					self._ax[i][j].set_xticklabels([])
					self._ax[i][j].set_yticklabels([])
			# Save and close the figure.
			if not os.path.isdir(_DEBUGDIR):
				os.mkdir(_DEBUGDIR)
			f = unicode(len(os.listdir(_DEBUGDIR))).zfill(8)
			self._fig.savefig(os.path.join(_DEBUGDIR, u'%s.jpg' % (f)))
			pyplot.close(self._fig)
			# Reset the variables.
			self._fig, self._ax = pyplot.subplots(nrows=2, ncols=3)
		
		return B
	
	
	def _process_eye_image(self, eyeimg, glint=True, mode='diameter', \
		debugindex=None):
		
		"""Finds the pupil and the glint in a single image of an eye, and
		returns the relevant parameters. Parameters that cannot be found
		will be None.
		
		Arguments
		
		eyeimg		-	A numpy.ndarray with unsigned, 8-bit
						integers that reflect the greyscale values
						of what is assumed to be an eye.
		
		Keyword Arguments
		
		glint			-	A Boolean that indicates whether the glint
						(the corneal reflection) should also be
						detected. Default = True.
		
		mode			-	A string that indicates how the pupil size
						should be reported.
						'diameter' reports the width of the rect in
						which the thresholded pupil fits
		
		Returns
		[px, py, ps, gx, gy]
					-	px is an integer indicating the pupil's
						likely horizontal position within the image.
						py is an integer indicating the pupil's
						likely vertical position within the image.
						ps is an integer indicating the pupil's
						size in pixels. The size reflects the width
						of the rect in which the pupil falls when
						mode=='diameter', or the total amount of
						thresholded pixels when mode'surface'.
						gx is an integer indicating the pupil's
						likely horizontal position within the image.
						gy is an integer indicating the pupil's
						likely vertical position within the image.
						All of these values can be None if they
						could not be obtained.
		"""
		
		# PUPIL
		# Create masks using the pupil (dark) and glint (light) luminance
		# thresholds.
		mask = {'p':eyeimg < self._pupt}
		if glint:
			# Create a mask using the pupil luminance threshold.
			mask['g'] = eyeimg > self._glit
		
		# Go through both masks, and save the extracted values.
		v = {'px':None, 'py':None, 'ps':None, 'gx':None, 'gy':None}
		for m in mask.keys():
	
			# Get connected components. The first component (comp==0) is
			# the background, and the others are connected components
			# without an ordering that's useful to us. The pupil will be
			# the largest component, so we will need to find that.
			comp, ncomp = ndimage.label(mask[m])

			# In DEBUG mode, draw the thresholded components.
			if self._debug and m == 'p':
				# Display the components in the central top/bottom pane.
				self._ax[debugindex][1].imshow(comp, cmap='jet')
				self._ax[debugindex][1].set_title("dark components")
			
			# Only proceed if there is more than one connected component
			# (if there is only one, that's just the background).
			if ncomp > 1:
				# Find the largest component (we assume this is the
				# pupil).
				pupcomp = 1
				compsum = 0
				for i in range(1, ncomp+1):
					s = numpy.sum(comp==i)
					if s >= compsum:
						compsum = s
						pupcomp = i
				# Fill in the gaps in the imperfect thresholding, to
				# better estimate the pupil area.
				pup = ndimage.binary_closing(comp==pupcomp).astype(int)
				# Get all points that are within the pupil as (y,x)
				# coordinates.
				coords = numpy.column_stack(numpy.nonzero(pup))
				# Extra check to see whether there are two or more
				# coordinates.
				if coords.shape[0] > 1:
					# Find the rect that encapsulates the pupil.
					x, y = coords[:,1].min(), coords[:,0].min()
					v['%ss' % (m)], h = coords[:,1].max() - x, \
						coords[:,0].max() - y
					v['%sx' % (m)] = x + v['%ss' % (m)]/2
					v['%sy' % (m)] = y + h/2
					# If required, calculate the pupil surface.
					if m == 'p' and mode == 'surface':
						v['%ss' % (m)] = numpy.sum(pup)

					# In DEBUG mode, colour the detected pupil, and draw a
					# draw a frame around it.
					if self._debug and m == 'p':
						# Create an RGB image from the grey image.
						img = numpy.dstack([eyeimg,eyeimg,eyeimg])
						# Colour the thresholded pupil blue.
						img[:,:,0][pup==1] = img[:,:,0][pup==1]/2
						img[:,:,1][pup==1] = img[:,:,1][pup==1]/2
						img[:,:,2][pup==1] = img[:,:,2][pup==1]/2 + 255/2
						# Display the detected pupil in the right
						# top/bottom pane.
						self._ax[debugindex][2].imshow(img)
						self._ax[debugindex][2].set_title("detected pupil")
						# Draw a green rectangle around the pupil.
						self._ax[debugindex][2].add_patch(patches.Rectangle( \
							(x-1,y-1), v['ps']+2, h+2, edgecolor=(0,1,0),
							fill=False, linewidth=1))
		
		return v['px'], v['py'], v['ps'], v['gx'], v['gy']
		