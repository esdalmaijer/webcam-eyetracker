import sys
import pygame
pygame.init()

BUFFSEP = 'edwinisdebeste'
CAMRES = 1024,768 # actually window size

# message
print("processing images...\n")

# open savefile
savefile = open('data/savefile.txt','r')

# read ALL contents in once, then close file again
raw = savefile.read()
savefile.close()
print("raw data is %d chars\n" % len(raw))

# split based on selected separator
raw = raw.split(BUFFSEP)
print("raw data is %d lines\n\n" % len(raw))

# wait for confirmation
#input("Press Enter to continue...")

# process strings and save image
framenr = 0
saveit = False
skipnext = False
for i in range(len(raw)-1):
	print("now processing frame %d, %d chars" % (i,len(raw[i])))
	if not skipnext:
		if sys.platform == 'win32':
			raw[i] = raw[i].replace("\n","")
		try:
			img = pygame.image.fromstring(raw[i],CAMRES,'RGB')
			saveit = True
		except:
			try:
				img = pygame.image.fromstring(raw[i]+'\n'+raw[i+1],CAMRES,'RGB')
				skipnext = True
				saveit = True
			except:
				print("dropped frame")
		if saveit:
			pygame.image.save(img,'data/frame%d.png' % framenr)
			saveit = False
			framenr += 1
	else:
		skipnext = False
