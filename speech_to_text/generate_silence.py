import sys
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) != 3:
        print("files not provided.")
        exit(1)

inputDir = sys.argv[1]
outputDir = sys.argv[2]

def augmentFillUpTo(data, sampleRate, seconds):
	L = seconds * sampleRate
	y_filled = data.copy()
	silenceLength = 0
	if len(y_filled) > L:
		diff = len(y_filled) - L
		silenceLength = -diff
		i = np.random.randint(0, diff)
		y_filled = y_filled[i:(i+L)]
	elif len(y_filled) < L:
		diff = L - len(y_filled)
		silenceLength = diff
		silence = np.zeros(diff)
		y_filled = np.concatenate((silence, y_filled))
	print("Adding silence = ", silenceLength)

	return y_filled

def makeSureDirectoryExists(fileName):
	directory = os.path.dirname(fileName)
	if not os.path.exists(directory):
		print("Creating directory: ", directory)
		os.makedirs(directory)

def saveWav(outputFile, data, sampleRate):
	print("Saving: ", outputFile)
	makeSureDirectoryExists(outputFile)
	librosa.output.write_wav(outputFile, data, sampleRate)

def generateSamples(filePath):
	data, sampleRate = librosa.load(filePath, sr=None)
	numberOfSamples = int(len(data) / sampleRate)
	fileName = os.path.basename(filePath)
	records = []
	for i in range(numberOfSamples - 1):
		start = i * sampleRate
		end = (i + 1) * sampleRate
		sample = data[start:end]
		sampleRecord = "silence/" + fileName.replace(".wav", "-%s.wav" % i)
		samplePath = outputDir + sampleRecord
		saveWav(samplePath, sample, sampleRate)
		records += [sampleRecord]
	sampleRecord = "silence/" + fileName.replace(".wav", "-%s.wav" % numberOfSamples)
	lastSamplePath = outputDir + sampleRecord
	start = numberOfSamples * sampleRate
	sample = augmentFillUpTo(data[start:], sampleRate, 1)
	saveWav(lastSamplePath, sample, sampleRate)
	records += [sampleRecord]
	return records

records = []
for file in os.listdir(inputDir):
	if file.endswith(".wav"):
		records += generateSamples(inputDir + file)

with open(outputDir + "silence_list.txt", "w") as silenceListFile:
	silenceListFile.writelines("%s\n" % l for l in records)


