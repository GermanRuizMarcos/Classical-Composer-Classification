'''
AUDIO CLASSICAL COMPOSER IDENTIFICATION BASED ON: 
A SPECTRAL BANDWISE FEATURE-BASED SYSTEM
'''
import essentia 
from essentia.standard import *
import glob
import numpy as np
import arff
from essentia.standard import *
from scipy import stats
import collections

# Dataset creation with specific attributes (spectral features) and a specific class (composer's name)

'''
Audio files trasformed into the frequency domain through a 1024-sample STFT with 50% overlap. 
The spectrum is divided into 50 mel-spaced bands.
'''

dirList = glob.glob("/home/usuario/Escritorio/SMC/2 term/Music Information Retrieval/Classical Composer Identification/datasets/bach/*.wav")
fft = FFT()
melbands = MelBands(numberBands = 50)

flatness = FlatnessDB()
rolloff = RollOff()
centroid = SpectralCentroidTime()
flux = Flux() 
energy = EnergyBand()
zero = ZeroCrossingRate()
spectrum = Spectrum()
w = Windowing(type = 'hann')
mfcc = MFCC()
silence = SilenceRate(thresholds = [0.01])


f = open('definitive_train.txt', 'wb')
f.write('@RELATION "composer dataset"\n')
f.write('\n')
f.write('@ATTRIBUTE filename STRING\n')
f.write('@ATTRIBUTE MFCC-0 REAL\n')
f.write('@ATTRIBUTE MFCC-1 REAL\n')
f.write('@ATTRIBUTE MFCC-2 REAL\n')
f.write('@ATTRIBUTE MFCC-3 REAL\n')
f.write('@ATTRIBUTE MFCC-4 REAL\n')
f.write('@ATTRIBUTE MFCC-5 REAL\n')
f.write('@ATTRIBUTE MFCC-6 REAL\n')
f.write('@ATTRIBUTE MFCC-7 REAL\n')
f.write('@ATTRIBUTE MFCC-8 REAL\n')
f.write('@ATTRIBUTE MFCC-9 REAL\n')
f.write('@ATTRIBUTE MFCC-10 REAL\n')
f.write('@ATTRIBUTE MFCC-11 REAL\n')
f.write('@ATTRIBUTE MFCC-12 REAL\n')
f.write('@ATTRIBUTE flatness-mean REAL\n')
f.write('@ATTRIBUTE flatness-variance REAL\n')
f.write('@ATTRIBUTE rolloff-mean REAL\n')
f.write('@ATTRIBUTE rolloff-variance REAL\n')
f.write('@ATTRIBUTE centroid-mean REAL\n')
f.write('@ATTRIBUTE centroid-variance REAL\n')
f.write('@ATTRIBUTE flux-mean REAL\n')
f.write('@ATTRIBUTE flux-variance REAL\n')
f.write('@ATTRIBUTE energy-mean REAL\n')
f.write('@ATTRIBUTE energy-variance REAL\n')
f.write('@ATTRIBUTE ZCR-mean REAL\n')
f.write('@ATTRIBUTE ZCR-variance REAL\n')
f.write('@ATTRIBUTE flatness-std REAL\n')
f.write('@ATTRIBUTE flatness-hmean REAL\n')
f.write('@ATTRIBUTE silences REAL\n')
f.write('@ATTRIBUTE composer {bach, beethoven, chopin, haydn, liszt, mendelssohn, mozart, vivaldi}\n')
f.write('\n')
f.write('@DATA\n')
		
for audio_file in dirList:

	flat = []
	rol = []
	cen = []
	flu = []
	ene = []
	zer = []
	mfccs = []
	stft = []
	sil = []

	# Loading audio
	audio = MonoLoader(filename = audio_file)()

	for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512, startFromZero=True):
			
		bands = melbands(spectrum(frame)) 
		stft.append(fft(frame))
		flat.append(flatness(bands))
		rol.append(rolloff(bands))
		cen.append(centroid(bands)) 
		flu.append(flux(bands))
		ene.append(energy(bands)) 
		zer.append(zero(frame))
		mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
		mfccs.append(mfcc_coeffs)
		sil.append(silence(frame))
	rate = collections.Counter()
	rate.update(sil)
	rate = rate.most_common(1)

	
	composer = 'bach'

	f.write('%s' %audio_file.split('/')[-1].split('.')[0].split('bach')[0])
	f.write(',')
	f.write('%r' %np.mean(mfccs[0]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[1]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[2]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[3]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[4]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[5]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[6]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[7]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[8]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[9]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[10]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[11]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[12]))
	f.write(',')
	f.write('%r' %np.mean(flat))
	f.write(',')
	f.write('%r' %np.var(flat))
	f.write(',')
	f.write('%r' %np.mean(rol))
	f.write(',')
	f.write('%r' %np.var(rol))
	f.write(',')
	f.write('%r' %np.mean(cen))
	f.write(',')
	f.write('%r' %np.var(cen))
	f.write(',')
	f.write('%r' %np.mean(flu))
	f.write(',')
	f.write('%r' %np.var(flu))
	f.write(',')
	f.write('%r' %np.mean(ene))
	f.write(',')
	f.write('%r' %np.var(ene))
	f.write(',')
	f.write('%r' %np.mean(zer))
	f.write(',')
	f.write('%r' %np.var(zer))
	f.write(',')
	f.write('%r' %np.std(flat))
	f.write(',')
	f.write('%r' %stats.hmean(flat))
	f.write(',')
	f.write('%r' %rate[0][1])
	f.write(',')
	f.write('%s' %composer)
	f.write('\n')
# 2

dirList = glob.glob("/home/usuario/Escritorio/SMC/2 term/Music Information Retrieval/Classical Composer Identification/datasets/beethoven/*.wav")

for audio_file in dirList:

	flat = []
	rol = []
	cen = []
	flu = []
	ene = []
	zer = []
	mfccs = []
	sil = []

	# Loading audio
	audio = MonoLoader(filename = audio_file)()

	for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512, startFromZero=True):
		
		bands = melbands(spectrum(frame)) 

		flat.append(flatness(bands))
		rol.append(rolloff(bands))
		cen.append(centroid(bands)) 
		flu.append(flux(bands))
		ene.append(energy(bands)) 
		zer.append(zero(frame))
		mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
		mfccs.append(mfcc_coeffs)
		sil.append(silence(frame))
	rate = collections.Counter()
	rate.update(sil)
	rate = rate.most_common(1)

	composer = 'beethoven'

	f.write('%s' %audio_file.split('/')[-1].split('.')[0].split('beethoven')[0])
	f.write(',')
	f.write('%r' %np.mean(mfccs[0]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[1]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[2]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[3]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[4]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[5]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[6]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[7]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[8]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[9]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[10]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[11]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[12]))
	f.write(',')
	f.write('%r' %np.mean(flat))
	f.write(',')
	f.write('%r' %np.var(flat))
	f.write(',')
	f.write('%r' %np.mean(rol))
	f.write(',')
	f.write('%r' %np.var(rol))
	f.write(',')
	f.write('%r' %np.mean(cen))
	f.write(',')
	f.write('%r' %np.var(cen))
	f.write(',')
	f.write('%r' %np.mean(flu))
	f.write(',')
	f.write('%r' %np.var(flu))
	f.write(',')
	f.write('%r' %np.mean(ene))
	f.write(',')
	f.write('%r' %np.var(ene))
	f.write(',')
	f.write('%r' %np.mean(zer))
	f.write(',')
	f.write('%r' %np.var(zer))
	f.write(',')
	f.write('%r' %np.std(flat))
	f.write(',')
	f.write('%r' %stats.hmean(flat))
	f.write(',')
	f.write('%r' %rate[0][1])
	f.write(',')
	f.write('%s' %composer)
	f.write('\n')
# 3

dirList = glob.glob("/home/usuario/Escritorio/SMC/2 term/Music Information Retrieval/Classical Composer Identification/datasets/chopin/*.wav")


for audio_file in dirList:

	flat = []
	rol = []
	cen = []
	flu = []
	ene = []
	zer = []
	mfccs = []
	sil = []

	# Loading audio
	audio = MonoLoader(filename = audio_file)()

	for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512, startFromZero=True):
		
		bands = melbands(spectrum(frame)) 

		flat.append(flatness(bands))
		rol.append(rolloff(bands))
		cen.append(centroid(bands)) 
		flu.append(flux(bands))
		ene.append(energy(bands)) 
		zer.append(zero(frame))
		mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
		mfccs.append(mfcc_coeffs)
		sil.append(silence(frame))
	rate = collections.Counter()
	rate.update(sil)
	rate = rate.most_common(1)

	composer = 'chopin'

	f.write('%s' %audio_file.split('/')[-1].split('.')[0].split('chopin')[0])
	f.write(',')
	f.write('%r' %np.mean(mfccs[0]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[1]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[2]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[3]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[4]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[5]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[6]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[7]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[8]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[9]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[10]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[11]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[12]))
	f.write(',')
	f.write('%r' %np.mean(flat))
	f.write(',')
	f.write('%r' %np.var(flat))
	f.write(',')
	f.write('%r' %np.mean(rol))
	f.write(',')
	f.write('%r' %np.var(rol))
	f.write(',')
	f.write('%r' %np.mean(cen))
	f.write(',')
	f.write('%r' %np.var(cen))
	f.write(',')
	f.write('%r' %np.mean(flu))
	f.write(',')
	f.write('%r' %np.var(flu))
	f.write(',')
	f.write('%r' %np.mean(ene))
	f.write(',')
	f.write('%r' %np.var(ene))
	f.write(',')
	f.write('%r' %np.mean(zer))
	f.write(',')
	f.write('%r' %np.var(zer))
	f.write(',')
	f.write('%r' %np.std(flat))
	f.write(',')
	f.write('%r' %stats.hmean(flat))
	f.write(',')
	f.write('%r' %rate[0][1])
	f.write(',')
	f.write('%s' %composer)
	f.write('\n')
# 4
dirList = glob.glob("/home/usuario/Escritorio/SMC/2 term/Music Information Retrieval/Classical Composer Identification/datasets/haydn/*.wav")

for audio_file in dirList:

	flat = []
	rol = []
	cen = []
	flu = []
	ene = []
	zer = []
	mfccs = []
	sil = []

	# Loading audio
	audio = MonoLoader(filename = audio_file)()

	for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512, startFromZero=True):
		
		bands = melbands(spectrum(frame)) 

		flat.append(flatness(bands))
		rol.append(rolloff(bands))
		cen.append(centroid(bands)) 
		flu.append(flux(bands))
		ene.append(energy(bands)) 
		zer.append(zero(frame))
		mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
		mfccs.append(mfcc_coeffs)
		sil.append(silence(frame))
	rate = collections.Counter()
	rate.update(sil)
	rate = rate.most_common(1)

	composer = 'haydn'

	f.write('%s' %audio_file.split('/')[-1].split('.')[0].split('haydn')[0])
	f.write(',')
	f.write('%r' %np.mean(mfccs[0]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[1]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[2]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[3]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[4]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[5]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[6]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[7]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[8]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[9]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[10]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[11]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[12]))
	f.write(',')
	f.write('%r' %np.mean(flat))
	f.write(',')
	f.write('%r' %np.var(flat))
	f.write(',')
	f.write('%r' %np.mean(rol))
	f.write(',')
	f.write('%r' %np.var(rol))
	f.write(',')
	f.write('%r' %np.mean(cen))
	f.write(',')
	f.write('%r' %np.var(cen))
	f.write(',')
	f.write('%r' %np.mean(flu))
	f.write(',')
	f.write('%r' %np.var(flu))
	f.write(',')
	f.write('%r' %np.mean(ene))
	f.write(',')
	f.write('%r' %np.var(ene))
	f.write(',')
	f.write('%r' %np.mean(zer))
	f.write(',')
	f.write('%r' %np.var(zer))
	f.write(',')
	f.write('%r' %np.std(flat))
	f.write(',')
	f.write('%r' %stats.hmean(flat))
	f.write(',')
	f.write('%r' %rate[0][1])
	f.write(',')
	f.write('%s' %composer)
	f.write('\n')
# 5
dirList = glob.glob("/home/usuario/Escritorio/SMC/2 term/Music Information Retrieval/Classical Composer Identification/datasets/liszt/*.wav")


for audio_file in dirList:

	flat = []
	rol = []
	cen = []
	flu = []
	ene = []
	zer = []
	mfccs = []
	sil = []

	# Loading audio
	audio = MonoLoader(filename = audio_file)()

	for frame in FrameGenerator(audio):
		
		bands = melbands(spectrum(frame)) 

		flat.append(flatness(bands))
		rol.append(rolloff(bands))
		cen.append(centroid(bands)) 
		flu.append(flux(bands))
		ene.append(energy(bands)) 
		zer.append(zero(frame))
		mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
		mfccs.append(mfcc_coeffs)
		sil.append(silence(frame))
	rate = collections.Counter()
	rate.update(sil)
	rate = rate.most_common(1)

	composer = 'liszt'

	f.write('%s' %audio_file.split('/')[-1].split('.')[0].split('liszt')[0])
	f.write(',')
	f.write('%r' %np.mean(mfccs[0]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[1]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[2]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[3]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[4]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[5]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[6]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[7]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[8]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[9]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[10]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[11]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[12]))
	f.write(',')
	f.write('%r' %np.mean(flat))
	f.write(',')
	f.write('%r' %np.var(flat))
	f.write(',')
	f.write('%r' %np.mean(rol))
	f.write(',')
	f.write('%r' %np.var(rol))
	f.write(',')
	f.write('%r' %np.mean(cen))
	f.write(',')
	f.write('%r' %np.var(cen))
	f.write(',')
	f.write('%r' %np.mean(flu))
	f.write(',')
	f.write('%r' %np.var(flu))
	f.write(',')
	f.write('%r' %np.mean(ene))
	f.write(',')
	f.write('%r' %np.var(ene))
	f.write(',')
	f.write('%r' %np.mean(zer))
	f.write(',')
	f.write('%r' %np.var(zer))
	f.write(',')
	f.write('%r' %np.std(flat))
	f.write(',')
	f.write('%r' %stats.hmean(flat))
	f.write(',')
	f.write('%r' %rate[0][1])
	f.write(',')
	f.write('%s' %composer)
	f.write('\n')
# 6
dirList = glob.glob("/home/usuario/Escritorio/SMC/2 term/Music Information Retrieval/Classical Composer Identification/datasets/mendelssohn/*.wav")

for audio_file in dirList:

	flat = []
	rol = []
	cen = []
	flu = []
	ene = []
	zer = []
	mfccs = []
	sil = []

	# Loading audio
	audio = MonoLoader(filename = audio_file)()

	for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512, startFromZero=True):

		bands = melbands(spectrum(frame)) 

		flat.append(flatness(bands))
		rol.append(rolloff(bands))
		cen.append(centroid(bands)) 
		flu.append(flux(bands))
		ene.append(energy(bands)) 
		zer.append(zero(frame))
		mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
		mfccs.append(mfcc_coeffs)
		sil.append(silence(frame))
	rate = collections.Counter()
	rate.update(sil)
	rate = rate.most_common(1)

	composer = 'mendelssohn'

	f.write('%s' %audio_file.split('/')[-1].split('.')[0].split('mendelssohn')[0])
	f.write(',')
	f.write('%r' %np.mean(mfccs[0]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[1]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[2]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[3]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[4]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[5]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[6]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[7]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[8]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[9]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[10]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[11]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[12]))
	f.write(',')
	f.write('%r' %np.mean(flat))
	f.write(',')
	f.write('%r' %np.var(flat))
	f.write(',')
	f.write('%r' %np.mean(rol))
	f.write(',')
	f.write('%r' %np.var(rol))
	f.write(',')
	f.write('%r' %np.mean(cen))
	f.write(',')
	f.write('%r' %np.var(cen))
	f.write(',')
	f.write('%r' %np.mean(flu))
	f.write(',')
	f.write('%r' %np.var(flu))
	f.write(',')
	f.write('%r' %np.mean(ene))
	f.write(',')
	f.write('%r' %np.var(ene))
	f.write(',')
	f.write('%r' %np.mean(zer))
	f.write(',')
	f.write('%r' %np.var(zer))
	f.write(',')
	f.write('%r' %np.std(flat))
	f.write(',')
	f.write('%r' %stats.hmean(flat))
	f.write(',')
	f.write('%r' %rate[0][1])
	f.write(',')
	f.write('%s' %composer)
	f.write('\n')
# 7
dirList = glob.glob("/home/usuario/Escritorio/SMC/2 term/Music Information Retrieval/Classical Composer Identification/datasets/mozart/*.wav")

for audio_file in dirList:

	flat = []
	rol = []
	cen = []
	flu = []
	ene = []
	zer = []
	mfccs = []
	sil = []

	# Loading audio
	audio = MonoLoader(filename = audio_file)()

	for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512, startFromZero=True):
		
		bands = melbands(spectrum(frame)) 

		flat.append(flatness(bands))
		rol.append(rolloff(bands))
		cen.append(centroid(bands)) 
		flu.append(flux(bands))
		ene.append(energy(bands)) 
		zer.append(zero(frame))
		mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
		mfccs.append(mfcc_coeffs)
		sil.append(silence(frame))
	rate = collections.Counter()
	rate.update(sil)
	rate = rate.most_common(1)

	composer = 'mozart'

	f.write('%s' %audio_file.split('/')[-1].split('.')[0].split('mozart')[0])
	f.write(',')
	f.write('%r' %np.mean(mfccs[0]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[1]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[2]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[3]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[4]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[5]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[6]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[7]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[8]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[9]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[10]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[11]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[12]))
	f.write(',')
	f.write('%r' %np.mean(flat))
	f.write(',')
	f.write('%r' %np.var(flat))
	f.write(',')
	f.write('%r' %np.mean(rol))
	f.write(',')
	f.write('%r' %np.var(rol))
	f.write(',')
	f.write('%r' %np.mean(cen))
	f.write(',')
	f.write('%r' %np.var(cen))
	f.write(',')
	f.write('%r' %np.mean(flu))
	f.write(',')
	f.write('%r' %np.var(flu))
	f.write(',')
	f.write('%r' %np.mean(ene))
	f.write(',')
	f.write('%r' %np.var(ene))
	f.write(',')
	f.write('%r' %np.mean(zer))
	f.write(',')
	f.write('%r' %np.var(zer))
	f.write(',')
	f.write('%r' %np.std(flat))
	f.write(',')
	f.write('%r' %stats.hmean(flat))
	f.write(',')
	f.write('%r' %rate[0][1])
	f.write(',')
	f.write('%s' %composer)
	f.write('\n')
# 8
dirList = glob.glob("/home/usuario/Escritorio/SMC/2 term/Music Information Retrieval/Classical Composer Identification/datasets/vivaldi/*.wav")

for audio_file in dirList:

	flat = []
	rol = []
	cen = []
	flu = []
	ene = []
	zer = []
	mfccs = []
	sil = []

	# Loading audio
	audio = MonoLoader(filename = audio_file)()

	for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512, startFromZero=True):
		
		bands = melbands(spectrum(frame)) 

		flat.append(flatness(bands))
		rol.append(rolloff(bands))
		cen.append(centroid(bands)) 
		flu.append(flux(bands))
		ene.append(energy(bands)) 
		zer.append(zero(frame))
		mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
		mfccs.append(mfcc_coeffs)
		sil.append(silence(frame))
	rate = collections.Counter()
	rate.update(sil)
	rate = rate.most_common(1)

	composer = 'vivaldi'

	f.write('%s' %audio_file.split('/')[-1].split('.')[0].split('vivaldi')[0])
	f.write(',')
	f.write('%r' %np.mean(mfccs[0]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[1]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[2]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[3]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[4]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[5]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[6]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[7]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[8]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[9]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[10]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[11]))
	f.write(',')
	f.write('%r' %np.mean(mfccs[12]))
	f.write(',')
	f.write('%r' %np.mean(flat))
	f.write(',')
	f.write('%r' %np.var(flat))
	f.write(',')
	f.write('%r' %np.mean(rol))
	f.write(',')
	f.write('%r' %np.var(rol))
	f.write(',')
	f.write('%r' %np.mean(cen))
	f.write(',')
	f.write('%r' %np.var(cen))
	f.write(',')
	f.write('%r' %np.mean(flu))
	f.write(',')
	f.write('%r' %np.var(flu))
	f.write(',')
	f.write('%r' %np.mean(ene))
	f.write(',')
	f.write('%r' %np.var(ene))
	f.write(',')
	f.write('%r' %np.mean(zer))
	f.write(',')
	f.write('%r' %np.var(zer))
	f.write(',')
	f.write('%r' %np.std(flat))
	f.write(',')
	f.write('%r' %stats.hmean(flat))
	f.write(',')
	f.write('%r' %rate[0][1])
	f.write(',')
	f.write('%s' %composer)
	f.write('\n')

f.write('%\n')
f.write('%\n')
f.write('%\n')

f.close()
	

