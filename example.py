import numpy as np
import matplotlib.pyplot as plt

import madmom
from madmom.audio import chroma
from madmom.features import chords
from madmom.audio import ffmpeg

print(madmom.audio.ffmpeg.get_file_info('data/yesterday.mp3'))


signal, sample_rate = madmom.audio.signal.load_audio_file('data/yesterday.mp3')

print(signal)
print(sample_rate)

spec = madmom.audio.spectrogram.Spectrogram('data/yesterday.mp3')

# calculate the difference
diff = np.diff(spec, axis=0)
# keep only the positive differences
pos_diff = np.maximum(0, diff)
# sum everything to get the spectral flux
sf = np.sum(pos_diff, axis=1)

plt.figure()
plt.imshow(spec[:, :200].T, origin='lower', aspect='auto')
plt.figure()
plt.imshow(pos_diff[:, :200].T, origin='lower', aspect='auto')
plt.figure()
plt.plot(sf)

plt.show()

dcp = chroma.DeepChromaProcessor()
chroma = dcp('data/yesterday.mp3')
print(chroma.shape)


plt.imshow(chroma[:100].T)
plt.show()

decode = chords.DeepChromaChordRecognitionProcessor()
print(decode(chroma))