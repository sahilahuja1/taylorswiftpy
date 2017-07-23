import numpy as np
import numpy as np
import matplotlib.pyplot as plt

import madmom
from madmom.audio import chroma
from madmom.features import chords
from madmom.audio import ffmpeg

dcp = chroma.DeepChromaProcessor()
chroma = dcp('data/yesterday.mp3')
np.savetxt('data/yesterday_chroma.txt', chroma, fmt='%f')

decode = chords.DeepChromaChordRecognitionProcessor()
decoded_chroma = decode(chroma)

np.savetxt('data/yesterday_decoded_chroma.txt', decoded_chroma, fmt='%f, %f, %s', delimiter=' ')