# -*- coding: utf-8 -*-

# Only run if this is the main thing.
if __name__ == u'__main__':

	import os
	import time
	
	from pygazetracker.webcam import WebCamTracker
	

	# # # # #
	# CONSTANTS
	
	# GENERAL
	DIR = os.path.dirname(os.path.abspath(__file__))
	LOGFILE = os.path.join(DIR, u'test')
	
	# PUPIL TRACKING
	FACEDETECT = True
	PUPTHRESHOLD = 50
	GLINTTHRESHOLD = 200
	GLINTDETECT = False
	PUPSIZEMODE = 'diameter'
	MINFACESIZE = (30,30)
	LEXPECT = (0.7,0.4)
	REXPECT = (0.3,0.4)
	MAXPUPDIST = 0.2
	MAXPUPSIZE = 0.3
	MAXCPU = 6
	
	# WEBCAM
	CAMNR = 0
	CAMMODE = 'B'
	
	# Initialise a new tracker instance.
	tracker = WebCamTracker(logfile=LOGFILE, facedetect=FACEDETECT, \
		pupthreshold=PUPTHRESHOLD, glintthreshold=GLINTTHRESHOLD, \
		glintdetect=GLINTDETECT, pupsizemode=PUPSIZEMODE, \
		minfacesize=MINFACESIZE, Lexpect=LEXPECT, Rexpect=REXPECT, \
		maxpupdist=MAXPUPDIST, maxpupsize=MAXPUPSIZE, maxcpu=MAXCPU, \
		camnr=CAMNR, mode=CAMMODE)
	
	# Wait for a bit, to allow the tracker to start initialise.
	time.sleep(3)
	
	# Start recording.
	tracker.start_recording()
	
	# Wait for three seconds.
	time.sleep(3)
	# Log something.
	tracker.log(u"This is a message!")
	# Wait for another three seconds.
	time.sleep(3)
	
	# Pause recording.
	tracker.stop_recording()
	
	# Close the connection with the tracker.
	tracker.close()
