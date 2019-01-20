import argparse
import sys
import os
import numpy as np
import time
import augmentation as aug
import helpers

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '-f', '--file',
    help='Text file containing on each line a path to a wav file to be processed. Each path has to be relative to parent dir of this file.')
parser.add_argument(
    '-o', '--outputdir',
    help='Directory where to save generated images.')
parser.add_argument(
    '-l', '--labelsfile',
    help='Text file containing on each line a label.')
args = parser.parse_args()

recordsFile = args.file
inputDir = os.path.dirname(recordsFile) + os.path.sep
outputDir = os.path.join(args.outputdir, "")

classes = helpers.readlines(args.labelsfile)

records = helpers.readlines(recordsFile);

for record in records:
	print("Processing: ", record)
	label, _ = record.split("/")
	print("label: %s, file: %s" % (label, record))
	if (label in classes):
		data, sampleRate = helpers.readwav(inputDir + record)
		for i in range(16):
			augmentedData = aug.applyAugmentation(data, sampleRate)
			print("data shape: ", augmentedData.shape)
			print("data: ", augmentedData)
			spectrogram = helpers.convertToLogMelSpectrogram(augmentedData, sampleRate)
			spectrogram = spectrogram * 64
			spectrogram = spectrogram.astype(np.uint8)
			outputFile = outputDir + record + "-%s.png" % time.time()
			helpers.saveSpectrogram(outputFile, spectrogram)

