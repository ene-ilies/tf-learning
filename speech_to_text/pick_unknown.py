import sys
import os
import random

if len(sys.argv) != 2:
        print("files not provided.")
        exit(1)

classes = ["yes", "no"]

inputFile = sys.argv[1]
inputDir = os.path.dirname(inputFile) + "/"

with open(inputFile, "r") as file:
	records = [line.rstrip('\n') for line in file]

i = 0
pickedRecords = []
while (i < 500):
	pickedRecord = records[random.randint(0, len(records) - 1)]
	label, _ = pickedRecord.split("/")
	if (not pickedRecord in pickedRecords) and (not label in classes):
		i += 1
		pickedRecords += [pickedRecord]

with open(inputDir + "unknown_list.txt", "w") as unknownListFile:
        unknownListFile.writelines("%s\n" % l for l in pickedRecords)
