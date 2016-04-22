# -*- coding: utf-8 -*-

import os


# # # # #
# CONSTANTS

# VERBOSITY
# 0 is errors only
# 1 is errors and warnings
# 2 is errors, warnings, and messages
# 3 is errors, warnings, messages, and debug info.
_VERBOSITY = 2

# DEBUG MODE
# In this mode, images of each step of the frame-processing are stored.
_DEBUG = False

# FILES AND FOLDERS
# Get the main directory
_DIR = os.path.abspath(os.path.dirname(__file__)).decode(u'utf-8')
# Get the directory for DEBUG images.
_DEBUGDIR = os.path.join(_DIR, 'DEBUG')
# Face detection cascade from:
# https://github.com/shantnu/FaceDetect
_FACECASCADE = os.path.join(_DIR, u'haarcascade_frontalface_default.xml')
# Eye detection cascade from:
# http://www-personal.umich.edu/~shameem/haarcascade_eye.html
_EYECASCADE = os.path.join(_DIR, u'haarcascade_eye.xml')


# # # # #
# INTERNAL FUNCTIONS

def _message(msgtype, sender, msg):
	
	if msgtype == u'error':
		raise Exception(u"ERROR in pygazetracker.%s: %s" \
			% (sender, msg))
	
	elif msgtype == u'warning' and _VERBOSITY >= 1:
		print(u"WARNING in pygazetracker.%s: %s" % (sender, msg))
	
	elif msgtype == u'message' and _VERBOSITY >= 2:
		print(u"MSG from pygazetracker.%s: %s" % (sender, msg))
	
	elif msgtype == u'debug' and _VERBOSITY >= 3:
		print(u"DEBUG pygazetracker.%s: %s" % (sender, msg))