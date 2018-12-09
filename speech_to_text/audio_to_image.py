import sys
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
import augmentation as aug

if len(sys.argv) != 4:
	print("files not provided.")
	exit(1)

recordsFile = sys.argv[1]
inputDir = os.path.dirname(recordsFile) + "/"
outputDir = sys.argv[2]
save = sys.argv[3] == "True"
augmentedOutputDir = "augmented/"

classes = ["yes", "no"]

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

with open(recordsFile, 'r') as f:
	records = [line.rstrip('\n') for line in f]

for record in records:
	print("Processing: ", record)
	label, _ = record.split("/")
	print("label: %s, file: %s" % (label, record))
	if save or (label in classes):
		data, sampleRate = librosa.load(inputDir + record, sr=None)
		for i in range(1):
			augmentedData = aug.applyAugmentation(data, sampleRate)
			print("data shape: ", augmentedData.shape)
			print("data: ", augmentedData)
			augmentedWavPath = augmentedOutputDir + record
			#saveWav(augmentedWavPath, augmentedData, sampleRate)
			#saveWav(augmentedWavPath + "orig.wav", data, sampleRate)
			spectrogram = convertToLogMelSpectrogram(augmentedData, sampleRate)
			#displaySpectrogram(spectrogram)
			#print("first: \n", spectrogram)
			#spectrogram = spectrogram[:,:,np.newaxis].astype(np.uint8)
			spectrogram = spectrogram * 64
			#print("second: \n", spectrogram)
			#displaySpectrogram(spectrogram)
			spectrogram = spectrogram.astype(np.uint8)
			#print("third: \n", spectrogram)
			#displaySpectrogram(spectrogram)
			outputFile = outputDir + record + "-%s.png" % time.time()
			saveSpectrogram(outputFile, spectrogram)
		break

