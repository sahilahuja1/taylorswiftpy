import numpy as np
import numpy as np
import matplotlib.pyplot as plt

import madmom
from madmom.audio import chroma
from madmom.features import chords
from madmom.audio import ffmpeg

text_file = open('covers/list1.list', 'r')
actuals = text_file.readlines()
text_file = open('covers/list2.list', 'r')
covers = text_file.readlines()

numsongs = len(actuals)

for i in range(numsongs):
	actuals[i] = 'covers/' + actuals[i][:-1] 
	covers[i] = 'covers/' + covers[i][:-1]

dcp = chroma.DeepChromaProcessor()

for i in range(numsongs):
	chromaActual = dcp(actuals[i] + '.mp3')
	chromaCover = dcp(covers[i] + '.mp3')

	np.savetxt(actuals[i] + '.txt', chromaActual, fmt='%f')
	np.savetxt(covers[i] + '.txt', chromaCover, fmt='%f')