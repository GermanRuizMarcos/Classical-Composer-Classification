import essentia
from essentia.standard import *
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

dirList = glob.glob("/home/usuario/Escritorio/SMC/2 term/Music Information Retrieval/Classical Composer Identification/datasets/bach/*.wav")

for audio_file in dirList:

	audio = MonoLoader(filename = audio_file)()
	naudio = np.array(audio)
	pic_name = str(audio_file.split('/')[-1].split('.')[0]) + '.png'
	
	dt = 30.0/(naudio.shape[0])
	t = np.arange(0.0, 30.0, dt)
	x = naudio  # the signal
	NFFT = 1024       # the length of the windowing segments
	Fs = int(1.0/dt)  # the sampling frequency


	ax1 = plt.subplot(211)
	#plt.plot(t, x)
	#plt.subplot(212, sharex=ax1)
	fig = plt.figure()
	Pxx, freqs, bins, im = plt.specgram(x, NFFT=NFFT, Fs=Fs, noverlap=900)
	#plt.show()
	fig.savefig(name)
	

