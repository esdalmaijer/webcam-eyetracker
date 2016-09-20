# -*- coding: utf-8 -*-

from __init__ import _message, _DEBUG, _DEBUGDIR, _EYECASCADE, _FACECASCADE

import time
from threading import Thread
from multiprocessing import cpu_count, Event, Lock, Process, Queue

import cv2
import numpy
from scipy import ndimage

# # # # #
# DEBUG MODE
# In DEBUG mode, create a Matplotlib figure.
if _DEBUG:
	_message(u'debug', u'generic', \
		u"DEBUG mode active; creating plots of each frame's processing steps.")
	import os
	from matplotlib import patches, pyplot
	global _FIG, _AX
	_FIG, _AX = pyplot.subplots(nrows=2, ncols=3)


# # # # # # # # # # # # # # #
# GENERIC EYE TRACKER CLASS #
# # # # # # # # # # # # # # #
#
# This class is intended to act as a parent class to specific implementations
# of eye tracking through different image sources. The routines are generic
# image processing routines that take care of the eye-tracking part of things,
# but the input of images needs to be implemented in a sub-class. For an
# example of such a sub-class, see pygazetracker.webcam.WebCamTracker.

class EyeTracker:
	
	"""Generic class that is intended to act as a parent class for other
	implementations. The general concept of processing images to find eyes
	in them is the same whether the input is a webcam stream, a video, or a
	series of images.
	"""
	
	def __init__(self, logfile=u'default', facedetect=True, eyedetect=True, \
		pupthreshold=50, glintthreshold=200, glintdetect=True, \
		pupsizemode=u'diameter', minfacesize=(30,30), Lexpect=(0.7,0.4), \
		Rexpect=(0.3,0.4), maxpupdist=0.2, maxpupsize=0.3, maxcpu=6, \
		**kwargs):
		
		"""Initialises an EyeTracker class.
		
		Keyword Arguments
		
		logfile		-	A string that indicates the path to the log
						file. An extension will be added
						automatically. Default = 'default'.
		
		facedetect		-	A Boolean that indicates whether face
						detection should be attempted before further
						processing (eye detection, and pupil/glint
						detection). Set this to False if you will
						be using the EyeTracker from close to an
						eye, in which cases faces need and could not
						be detected. Default = True.
		
		pupthreshold	-	An integer that indicates what the highest
						luminance value is that is still considered
						to be part of the pupil. This value needs to
						be between 0 and 255. Default = 50.
		
		glintthreshold	-	An integer that indicates what the lowest
						luminance value is that is still considered
						to be part of the glint. This value needs to
						be between 0 and 255. Default = 200.
		
		glintdetect		-	A Boolean that indicates whether the glint
						(the corneal reflection) should also be
						detected. Default = True.
		
		pupsizemode		-	A string that indicates how the pupil size
						should be reported.
						'diameter' reports the width of the rect in
						which the thresholded pupil fits.
						'surface' reports the number of thresholded
						pixels that are assumed to be the pupil.
		
		minfacesize		-	A (w,h) tuple that indicates what size a
						face should minimally be. Default = (30,30)
		
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
		
		maxpupdist		-	A float that indicates what the maximal
						allowable distance is between the expected
						eye position, and the position of detected
						potential eye. The maximal distance is
						defined as a proportion of the image height.
						It can also be set to None. Default = (0.2)
		
		maxpupsize		-	A float that indicates what the maximal
						allowable width is of the detected eyes. The
						maximal size is defined as a proportion of
						the image width. It can also be set to None.
						Default = (0.3)

		maxcpu		-	Integer indicating the maximum amount of
						parallel processes that will be doing all
						of the image processing. This happens in
						parallel to speed things up; the processing
						time on one CPU can't keep up with the
						camera's sampling rate. Default = 6.
		"""

		# DEBUG message.
		_message(u'debug', u'generic.EyeTracker.__init__', \
			u"Initialising a new EyeTracker.")
		
		# GENERAL SETTINGS
		# Face detection yes/no, and from what size.
		self._facedetect = facedetect
		self._minfacesize = minfacesize
		# Face eye yes/no.
		self._eyedetect = eyedetect
		# Eye detection settings. These are relative positions of where
		# each eye is expected to be in a frame, how far away detected eyes
		# are allowed to be from the expected locations, and how large the
		# detected eyes are allowed to be. (All defined as proportions of
		# the frame's width and height.)
		self._Lexpect = Lexpect
		self._Rexpect = Rexpect
		self._maxpupdist = maxpupdist
		self._maxpupsize = maxpupsize
		# Pupil detection thresholds (dark for pupil, bright for glint),
		# and additional options that determine whether glints should be
		# detected, and how the pupil size should be reported.
		self._pupt = pupthreshold
		self._glit = glintthreshold
		self._glintdetect = glintdetect
		self._pupsizemode = pupsizemode
		
		# ALIVE EVENT
		# This event signals whether the tracker is still alive. It should
		# only be cleared when closing the connection to the tracker!
		self._alive = Event()
		self._alive.set()
		
		# FRAME OBTAINING THREAD
		# Boolean that turns to True when a connection with the source of
		# frames has been established.
		self._connected = False
		# We need a Queue for frames that are generated in the obtainer
		# Thread. The Queue is read out by the parallel processes.
		self._framequeue = Queue()
		# We need a lock to prevent potential simultaneous attempts to
		# access the image source at the same time. This shouldn't actually
		# be possible in the current implementation, but may be added in
		# the future.
		self._sourcelock = Lock()
		# Start the frame obtaining Thread
		_message(u'debug', u'generic.EyeTracker.__init__', \
			u"Starting a Thread to obtain frames.")
		self._frame_obtainer_thread = Thread(target=self._frame_obtainer, \
			args=[self._alive, self._framequeue])
		self._frame_obtainer_thread.name = u'frame_obtainer'
		self._frame_obtainer_thread.daemon = True
		self._frame_obtainer_thread.start()

		# PARALLEL PROCESSING
		# We need a Queue for samples that are generated in the parallel
		# processes that are simultaneously processing new frames.
		self._samplequeue = Queue()
		# Check how many CPUs we can use.
		cpus = cpu_count()
		if cpus > maxcpu:
			cpus = maxcpu
		# Start parallel processes to do image processing.
		_message(u'debug', u'generic.EyeTracker.__init__', \
			u"Starting %d parallel processes to process frames into samples." \
			% (cpus-1))
		self._frame_processes = []
		for i in range(1, cpus):
			p = Process(target=_frame_processer, \
				args=[self._alive, self._framequeue, self._samplequeue, \
				self._pupt, self._glit, self._facedetect, self._eyedetect, \
				self._minfacesize, self._Lexpect, self._Rexpect, \
				self._maxpupdist, self._maxpupsize, self._glintdetect, \
				self._pupsizemode])
			p.name = u'frame_processor_%d' % (i)
			p.daemon = True
			p.start()
			self._frame_processes.append(p)
		
		# SAMPLE WRITING
		# Variable that keeps track of the latest sample.
		self._latest_sample = [0, numpy.zeros((2,5))*numpy.NaN]
		# Boolean that signals whether the recording Thread should be
		# active or not.
		self._recording = False
		# Lock to prevent simultaneous access to the log file.
		self._loglock = Lock()
		# The log file is an open text file. It will be opened when
		# self._start_recording is called, and it will be closed when
		# self._stop_recording is called. Between calling those two
		# functions, samples will be appended to the log. To prevent
		# samples from being appended to an existing log file, here we
		# open a new logfile with in 'w' mode, thereby erasing any existing
		# content of a previous log file. This means users need to be
		# careful when naming their files, to prevent overwriting.
		self._logfilename = u'%s.tsv' % (logfile)
		_message(u'debug', u'generic.EyeTracker.__init__', \
			u"Creating new logfile '%s'." \
			% (self._logfilename))
		# Create a header for the log file.
		l = [u'time']
		l.extend([u'Lpx', u'Lpy', u'Lps', u'Lgx', u'Lgy'])
		l.extend([u'Rpx', u'Rpy', u'Rps', u'Rgx', u'Rgy'])
		line = u'\t'.join(map(unicode, l)) + u'\n'
		# Create a new log file.
		self._loglock.acquire(True)
		self._logfile = open(self._logfilename, u'w')
		self._logfile.write(line)
		self._logfile.close()
		self._loglock.release()

		# Start the sample logging Thread
		_message(u'debug', u'generic.EyeTracker.__init__', \
			u"Starting a Thread to log samples to file '%s'." \
			% (self._logfilename))
		self._sample_logging_thread = Thread(target=self._sample_logger, \
			args=[self._alive, self._samplequeue])
		self._sample_logging_thread.name = u'sample_logger'
		self._sample_logging_thread.daemon = True
		self._sample_logging_thread.start()
		
		# CUSTOM INITIALISATION
		# Run the custom initialisation procedure.
		self.connect(**kwargs)
	
	
	def close(self, **kwargs):
		
		"""Use this function to implement the ending of a specific type
		of eye tracking.
		"""
		
		# Only close the connection EyeTracker if it isn't closed yet.
		if self._connected:

			# Signal the _frame_obtainer that the tracker isn't connected
			# anymore.
			self._connected = False
			
			# Signal all Threads and Processes that they should stop.
			self._alive.clear()
			
			# Wait for all frame-processing Processes to join.
			for p in self._frame_processes:
				p.join()
			
			# Wait for the frame-obtaining Thread to join.
			self._frame_obtainer_thread.join()
			
			# Wait for the logging Thread to join.
			self._sample_logging_thread.join()
			
			# Call the custom closing implementation.
			self._close(**kwargs)
	
	
	def connect(self, **kwargs):
		
		"""Use this function to implement the initialisation of a specific
		type of eye tracking.
		"""
		
		# Only intialise the EyeTracker if it isn't initialised yet.
		if not self._connected:
			
			# Signal the frame obtainer that we are connected!
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
	
	
	def sample(self):
		
		"""Returns the newest sample.
		
		Returns

		t, [L, R]		-	t is a timestamp associated with the latest
						sample.
						[L, R] is a NumPy array that contains the
						following:
						L = numpy.array([px, py, ps, gx, gy])
						R = numpy.array([px, py, ps, gx, gy])
						px is an integer indicating the pupil's
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
		
		# Return the latest sample.
		# TODO: Potential issue here is that this sample is at least five
		# samples old. Maybe we could peek ahead in the Queue, and actually
		# get the latest?
		return self._latest_sample
	
	
	def log(self, msg):
		
		"""Logs a message to the log file.
		
		Arguments
		
		msg			-	A string that should be added to the log
						file.
		"""
		
		# Make sure that the message is in a unicode format.
		msg = msg.decode(u'utf-8')
		
		# A message is a time-stamped sample, with a slightly different
		# content than normal samples. In essence, the log message sample
		# is pretending to be a real sample, with a list for the 'left'
		# and the 'right' eye. It seems a bit ridiculous, but the nested
		# lists are there to prevent the strings from being pulled apart
		# when self._log_sample tries to turn them into lists.
		sample = [time.time(), [[u'MSG'], [msg]]]

		# Put the sample in the Queue.
		self._samplequeue.put(sample)


	def start_recording(self):
		
		"""Starts the writing of samples to the log file.
		"""

		# Only start recording if it isn't currently active.
		if not self._recording:
			_message(u'debug', u'generic.EyeTracker.__init__', \
				u"Starting recording, and re-opening logfile '%s'." \
				% (self._logfilename))
			# Signal the recording thread to start.
			self._recording = True
			# Re-open the logfile.
			self._loglock.acquire(True)
			self._logfile = open(self._logfilename, u'a')
			self._loglock.release()


	def stop_recording(self):
		
		"""Pauses the writing of samples to the log file.
		"""

		# Only pause recording if recording is currently active.
		if self._recording:
			# Signal the recording Thread to stop what it's doing.
			self._recording = False
			# Wait for a bit, to allow the emptying of the local queue.
			time.sleep(0.2)
			# Close the logfile.
			self._loglock.acquire(True)
			self._logfile.close()
			self._loglock.release()
			_message(u'debug', u'generic.EyeTracker.__init__', \
				u"Stopped recording, and closed logfile '%s'" \
				% (self._logfilename))


	def _get_frame(self):
		
		"""Use this function to implement how the EyeTracker should obtain
		a frame. A frame is supposed to be a two-dimensional image, usually
		an image from a webcam or a video that was converted to greyscale.
		Note that a frame should be a numpy.ndarray with shape=(w,h), and
		dtype='uint8'. In addition to the frame, a success Boolean should
		be returned by this function. It tells the functions that call
		_get_frame whether a new frame was available. (See below what the
		returned values should be exactly.)
		
		IMPORTANT: This function should not have any keyword arguments.
		Any settings should be handled through properties of self.
		
		Returns
		
		success, frame	-	success is a Boolean that indicates whether
						a frame could be obtained.
						frame is a numpy.ndarray with unsigned,
						8-bit integers that reflect the greyscale
						values of the image. If no frame could be
						obtained, None will be returned.
		"""
		
		# Obtain a frame.
		_message(u'message', u'generic.EyeTracker._get_frame', \
			u"Implement your own _get_frame functionality")
		
		return False, None

	
	def _close(self, **kwargs):
		
		"""Use this function to implement the specifics of closing a
		connection in your eye-tracking implementation. You could, for
		example, use it to close the connection to a webcam. This function
		is automatically called when the close() method is passed, and this
		setup allows you to pass your own keyword arguments to close (which
		will then be passed on to _close).
		"""
		
		# CUSTOM IMPLEMENTATION HERE
		_message(u'message', u'generic.EyeTracker._close', \
			u"Implement your own _close functionality")

	
	# # # # # #
	# HELPERS #
	# # # # # #
	
	def _log_sample(self, sample):
		
		"""Writes a sample to the log file.
		
		Arguments

		sample		-	A (t, [L, R]) tuple/list. Explanation below:
		t, [L, R]		-	t is a timestamp associated with the sample.
						[L, R] is a NumPy array that contains the
						following:
						L = numpy.array([px, py, ps, gx, gy])
						R = numpy.array([px, py, ps, gx, gy])
						px is an integer indicating the pupil's
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
		
		# Compose a string that can be written to the log file. This
		# contains the timestamp in milliseconds (hence *1000), the left
		# eye's values, and the right eye's values, all separated by tabs.
		l = [int(sample[0]*1000)]
		l.extend(list(sample[1][0]))
		l.extend(list(sample[1][1]))
		line = u'\t'.join(map(unicode, l)) + u'\n'

		# Write to the log file. Acquire the log lock first, and release it
		# after writing, to prevent simultaneous writing to the same log.
		self._loglock.acquire(True)
		self._logfile.write(line)
		self._loglock.release()

	
	# # # # # # # # # # #
	# RUNNING PROCESSES #
	# # # # # # # # # # #
	
	def _frame_obtainer(self, event, framequeue):
		
		"""Continuously tries to get new frames using self._get_frame, and
		puts the obtained frames in the framequeue.
		"""
		
		# Continuously run.
		while event.is_set():
			
			# Only try to get frames if a connection is established.
			if self._connected:
				
				# Obtain a new frame, and an associated timestamp.
				# Make sure to lock the source while obtaining the
				# frame, to prevent simultaneous access. (This
				# shouldn't actually happen for the moment, but might
				# become a feature in the future.)
				self._sourcelock.acquire(True)
				success, frame = self._get_frame()
				t = time.time()
				self._sourcelock.release()
				
				# Put the obtained timestamp and frame in the queue.
				framequeue.put([t, frame])
			
			# If the tracker is not connected, wait for a bit to avoid
			# wasting processing resources on continuously checking
			# whether the tracker is already connected.
			else:
				# Pause for 10 milliseconds.
				time.sleep(0.01)
	
	
	def _sample_logger(self, event, samplequeue):
		
		"""Continuously monitors the Queue, and writes samples to the log
		file whenever over five are still in the Queue.
		"""
		
		# Create a list to keep track of samples.
		timestamps = []
		samplelist = []
		
		# Continuously run.
		while event.is_set():
			
			# Only process samples if the tracker is recording.
			if self._recording:
				
				# Obtain a new sample if the Queue isn't empty, and lock
				# the Queue while we're using it.
				if not samplequeue.empty():
#				if samplequeue.qsize() > 0:
					# Get the oldest sample in the Queue.
					sample = samplequeue.get()
					# Add the sample to the list.
					timestamps.append(sample[0])
					samplelist.append(sample[1])
					# Store the sample locally, but only if it's not
					# a message.
					if sample[1][0][0] != 'MSG':
						self._latest_sample = sample
				
				# Write the oldest samples from the list, but make sure
				# there are always at least five samples still in the
				# list. We do this, because the sampling happens
				# asynchronously in several parallel processes. This
				# might result in newer samples being processed and
				# becoming available before older samples. Obviously,
				# we want to log the samples in chronological order of
				# obtaining them, not of them becoming available. So
				# we keep a buffer of five samples, which should
				# hopefully be enough to allow slightly older samples
				# to come in.
				while len(timestamps) > 5:
					# Find the oldest timestamp.
					i = numpy.argmin(timestamps)
					t = timestamps.pop(i)
					LR = samplelist.pop(i)
					# Log the sample.
					self._log_sample([t, LR])
			
			# If we're not recording anymore, but there are samples left
			# in the list, then we need to process those first.
			elif not self._recording and len(timestamps) > 0:
				# Empty out the sample buffer.
				while len(timestamps) > 0:
					# Find the oldest timestamp.
					i = numpy.argmin(timestamps)
					t = timestamps.pop(i)
					LR = samplelist.pop(i)
					# Log the sample.
					self._log_sample([t, LR])
			
			# If the tracker is not recording, wait for a bit to avoid
			# wasting processing resources on continuously checking
			# whether the tracker is recording.
			else:
				# Pause for 10 milliseconds.
				time.sleep(0.01)

	
# # # # # # # # # # #
# IMAGE PROCESSING  #
# # # # # # # # # # #
#
# These functions are defined as functions rather than methods of EyeTracker,
# because they need to be picklable in order to be run in parallel processes.
# Parallel processing is crucial here to keep up the sampling rate, which is
# now dependent on how quick we can poll the webcam, but it would otherwise
# depend on how quick we can poll the webcam AND how quick we could then
# process the webcam's image.

def _frame_processer(event, framequeue, samplequeue, pupthreshold, \
	glintthreshold, facedetect, eyedetect, minfacesize, Lexpect, Rexpect, \
	maxpupdist, maxpupsize, glintdetect, pupsizemode):
	
	"""Continuously obtains frames from the framequeue, processes them into
	samples, and puts those samples in the samplequeue.
	
	Arguments
	
	framequeue		-	A multiprocessing.Queue instance that contains
					frames. The _frame_processer will check if any
					new frames are available from the framequeue, and
					it will take them out.
					
	
	samplequeue		-	A multiprocessing.Queue instance that is used by
					the _frame_processer to put samples in.
	"""

	# CASCADING ALGORITHMS FOR FACE AND EYE DETECTION
	# Initialise the cascading algorithms for face and eye detection.
	# NOTE: Credit for and an explanation of the following classifiers
	# can be found on OpenCV's documentation website. Please see:
	# http://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html
	face_cascade = cv2.CascadeClassifier(_FACECASCADE)
	eye_cascade = cv2.CascadeClassifier(_EYECASCADE)
	
	# Continuously run.
	while event.is_set():

		# Only look for frames if the Queue isn't empty.
		if not framequeue.empty():

			# Get a new frame from the frame queue. The Queue should
			# handle locking internally, so we don't have to worry about
			# simultaneous attempts to access the Queue by processes.
			t, frame = framequeue.get()
		
			# Process the frame to get a sample.
			LR = _get_sample(frame, pupthreshold, glintthreshold, \
				face_cascade, eye_cascade, facedetect, eyedetect, \
				minfacesize, Lexpect, Rexpect, maxpupdist, maxpupsize, \
				glintdetect, pupsizemode)
			
			# Put the sample in the Queue. The Queue should
			# handle locking internally, so we don't have to worry about
			# simultaneous attempts to access the Queue by processes.
			samplequeue.put([t, LR])


def _get_sample(frame, pupthreshold, glintthreshold, face_cascade, \
	eye_cascade, facedetect, eyedetect, minfacesize, Lexpect, Rexpect, \
	maxpupdist, maxpupsize, glintdetect, pupsizemode):
	
	"""Gets a new frame from the input, and processes that through the
	standard pipeline of optional face detection, eye detection, and
	pupil and optional glint detection.
	
	Returns

	t, [L, R]		-	t is a timestamp associated with the sample.
					[L, R] is a NumPy array that contains the
					following:
					L = numpy.array([px, py, ps, gx, gy])
					R = numpy.array([px, py, ps, gx, gy])
					px is an integer indicating the pupil's
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
	
	# Optionally do face detection.
	if facedetect:
		success, frame = _crop_face(frame, face_cascade, minfacesize)
	# Optionally detect the eye(s) in a frame
	if eyedetect:
		success, eyes = _crop_eyes(frame, eye_cascade, Lexpect=Lexpect, \
			Rexpect=Rexpect, maxdist=maxpupdist, maxsize=maxpupsize)
	else:
		eyes = [frame, None]
	# Detect the pupils (and optionally the glint) in the eyes.
	LR = _find_pupils(eyes[0], eyes[1], pupthreshold=pupthreshold, \
		glintthreshold=glintthreshold, glint=glintdetect, mode=pupsizemode)
	
	return LR


def _crop_face(frame, face_cascade, minsize=(30, 30)):
	
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
	
	# Return straight away when frame==None
	if frame is None:
		return False, None
	
	# Find all potential faces in the frame.
	faces = face_cascade.detectMultiScale(
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
	if _DEBUG:
		# Display the frame in the top-left pane.
		_AX[0][0].set_title("potential faces")
		_AX[0][0].imshow(frame, cmap='gray')
		for i in range(len(faces)):
			r = faces[i]
			_AX[0][0].add_patch(patches.Rectangle( \
				(r[0],r[1]), r[2], r[3], fill=False, linewidth=1))
		_AX[0][0].add_patch(patches.Rectangle( \
			(x,y), w, h, fill=False, linewidth=3))
	
	# Return the cropped frame.
	return success, frame[y:y+h, x:x+w]


def _crop_eyes(facecrop, eye_cascade, Lexpect=(0.7,0.4), Rexpect=(0.3,0.4), \
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
	
	# Return straight away when facecrop==None
	if facecrop is None:
		return False, [None, None]
	
	# DETECT THE EYES
	eyes = eye_cascade.detectMultiScale(facecrop)
	
	# Return if no eyes could be detected.
	if len(eyes) == 0:
		return False, [None, None]
	
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
	if (li is None) & (ri is None):
		success = False
	# If the left eye was detected, crop it from the face.
	if li is None:
		left = None
	else:
		x, y, w, h = eyes[li]
		y += h/4
		h /= 2
		left = facecrop[y:y+h, x:x+w]
	# If the right eye was detected, crop it from the face.
	if ri is None:
		right = None
	else:
		x, y, w, h = eyes[ri]
		y += h/4
		h /= 2
		right = facecrop[y:y+h, x:x+w]

	# In DEBUG mode, draw the original frame and the detected eyes.
	if _DEBUG:
		# Display the face in the bottom-left pane.
		_AX[1][0].imshow(facecrop, cmap='gray')
		_AX[1][0].set_title("potential eyes")
		for i in range(len(eyes)):
			r = eyes[i]
			if i == li or i == ri:
				_AX[1][0].add_patch(patches.Rectangle( \
					(r[0],r[1]), r[2], r[3], fill=False, \
					linewidth=3))
			else:
				_AX[1][0].add_patch(patches.Rectangle( \
					(r[0],r[1]), r[2], r[3], fill=False, \
					linewidth=1))

	return success, [left, right]


def _find_pupils(left, right, pupthreshold=50, glintthreshold=200, \
	glint=True, mode='diameter'):
	
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
					which the thresholded pupil fits.
					'surface' reports the number of thresholded
					pixels that are assumed to be the pupil.
	
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
					thresholded pixels when mode=='surface'.
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
		if eyeimg is None:
			B[i,:] *= numpy.NaN
		else:
			B[i,:] = _process_eye_image(eyeimg, \
				pupthreshold=pupthreshold, glintthreshold=glintthreshold, \
				glint=True, mode=mode, debugindex=i)
		i += 1
	
	# In DEBUG mode, save and reset the image.
	if _DEBUG:
		global _FIG, _AX
		# Remove all axes' ticklabels
		for i in range(len(_AX)):
			for j in range(len(_AX[i])):
				_AX[i][j].set_xticklabels([])
				_AX[i][j].set_yticklabels([])
		# Save and close the figure.
		if not os.path.isdir(_DEBUGDIR):
			os.mkdir(_DEBUGDIR)
		f = unicode(len(os.listdir(_DEBUGDIR))).zfill(8)
		_FIG.savefig(os.path.join(_DEBUGDIR, u'%s.jpg' % (f)))
		pyplot.close(_FIG)
		# Reset the variables. (The global bit is there to tell Python that
		# _FIG and _AX do not refer to a local definition, but instead
		# refer to the global _FIG and _AX that are defined outside of this
		# function.)
		_FIG, _AX = pyplot.subplots(nrows=2, ncols=3)
	
	return B


def _process_eye_image(eyeimg, pupthreshold=50, glintthreshold=200, \
	glint=True, mode='diameter', debugindex=None):
	
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
	
	# Create masks using the pupil (dark) and glint (light) luminance
	# thresholds.
	mask = {'p':eyeimg < pupthreshold}
	if glint:
		# Create a mask using the pupil luminance threshold.
		mask['g'] = eyeimg > glintthreshold
	
	# Go through both masks, and save the extracted values.
	v = {'px':None, 'py':None, 'ps':None, 'gx':None, 'gy':None}
	for m in mask.keys():

		# Get connected components. The first component (comp==0) is
		# the background, and the others are connected components
		# without an ordering that's useful to us. The pupil will be
		# the largest component, so we will need to find that.
		comp, ncomp = ndimage.label(mask[m])

		# In DEBUG mode, draw the thresholded components.
		if _DEBUG and m == 'p':
			# Display the components in the central top/bottom pane.
			_AX[debugindex][1].imshow(comp, cmap='jet')
			_AX[debugindex][1].set_title("dark components")
		
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
				if _DEBUG and m == 'p':
					# Create an RGB image from the grey image.
					img = numpy.dstack([eyeimg,eyeimg,eyeimg])
					# Colour the thresholded pupil blue.
					img[:,:,0][pup==1] = img[:,:,0][pup==1]/2
					img[:,:,1][pup==1] = img[:,:,1][pup==1]/2
					img[:,:,2][pup==1] = img[:,:,2][pup==1]/2 + 255/2
					# Display the detected pupil in the right
					# top/bottom pane. (Only when the glint is not
					# to be marked.)
					if not glint:
						_AX[debugindex][2].imshow(img)
						_AX[debugindex][2].set_title("detected pupil")
						# Draw a green rectangle around the pupil.
						_AX[debugindex][2].add_patch(patches.Rectangle( \
							(x-1, y-1), v['ps']+2, h+2, \
							edgecolor=(0,1,0), fill=False, linewidth=1))
				# In DEBUG mode, colour the detected glint, and draw a
				# draw a frame around it.
				if _DEBUG and glint and m == 'g':
					# Colour the thresholded glint red.
					img[:,:,0][pup==1] = img[:,:,0][pup==1]/2 + 255/2
					img[:,:,1][pup==1] = img[:,:,1][pup==1]/2
					img[:,:,2][pup==1] = img[:,:,2][pup==1]/2
					# Display the detected pupil and glint in the
					# right top/bottom pane.
					_AX[debugindex][2].imshow(img)
					_AX[debugindex][2].set_title("detected pupil")
					# Draw a green rectangle around the pupil.
					_AX[debugindex][2].add_patch(patches.Rectangle( \
						(v['px']-v['ps']/2-1, v['py']-v['ps']/2-1), \
						v['ps']+2, v['ps']+2, edgecolor=(0,1,0),
						fill=False, linewidth=1))
	
	return v['px'], v['py'], v['ps'], v['gx'], v['gy']
