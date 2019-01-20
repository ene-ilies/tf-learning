import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image

def readwav(filename):
	return librosa.load(filename, sr=None)

def readlines(filename):
	with open(filename, 'r') as f:
		return [line.rstrip('\n') for line in f]

def convertToLogMelSpectrogram(data, sampleRate):
	melSpectrogram = librosa.feature.melspectrogram(y=data, sr=sampleRate, n_mels=128, hop_length=252, power=2.0)
	return melSpectrogram
	#return librosa.power_to_db(melSpectrogram,
         #       ref=np.max)

def displaySpectrogram(spectrogram):
	plt.figure(figsize=(10, 4))
	librosa.display.specshow(spectrogram,
		y_axis='mel', fmax=8000,
		x_axis='time')
	plt.colorbar(format='%+2.0f dB')
	plt.title('Mel spectrogram')
	plt.tight_layout()
	plt.show()

def makeSureDirectoryExists(fileName):
	directory = os.path.dirname(fileName)
	if not os.path.exists(directory):
		print("Creating directory: %s", directory)
		os.makedirs(directory)

def saveSpectrogram(outputFile, spectrogram):
	makeSureDirectoryExists(outputFile)
	print("Saving spectrogram to: %s, with size: %s" % (outputFile, str(spectrogram.shape)))
	Image.fromarray(spectrogram).save(outputFile)

def saveWav(outputFile, data, sampleRate):
	makeSureDirectoryExists(outputFile)
	print("Saving to: %s, at sampleRate: %s" % (outputFile, sampleRate) )
	librosa.output.write_wav(outputFile, data, sampleRate)
