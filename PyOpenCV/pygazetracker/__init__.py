# -*- coding: utf-8 -*-

import os

# FILES AND FOLDERS
# Get the main directory
DIR = os.path.abspath(os.path.dirname(__file__)).decode(u'utf-8')
# Face detection cascade from:
# https://github.com/shantnu/FaceDetect
_FACECASCADE = os.path.join(DIR, u'haarcascade_frontalface_default.xml')
# Eye detection cascade from:
# http://www-personal.umich.edu/~shameem/haarcascade_eye.html
_EYECASCADE = os.path.join(DIR, u'haarcascade_eye.xml')
