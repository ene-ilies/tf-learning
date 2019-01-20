#!/usr/bin/env python3
"""Create a recording with arbitrary duration.

PySoundFile (https://github.com/bastibe/PySoundFile/) has to be installed!

"""
import argparse
import tempfile
import queue
import sys
import librosa
import numpy as np
from inferencer import SpeechToTextInferencer
import helpers

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '-ld', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-r', '--samplerate', type=int, help='sampling rate')
parser.add_argument(
    '-tr', '--targetsamplerate', type=int, default=16000, help='target sampling rate')
parser.add_argument(
    '-c', '--channels', type=int, default=1, help='number of input channels')
parser.add_argument(
    'filename', nargs='?', metavar='FILENAME',
    help='audio file to store recording to')
parser.add_argument(
    '-t', '--subtype', type=str, help='sound file subtype (e.g. "PCM_24")')
parser.add_argument(
    '-cp', '--checkpointdir',
    help='Directory where to read checkpoint file from.')
parser.add_argument(
    '-l', '--labelsfile',
    help='Text file containing on each line a label.')
args = parser.parse_args()

currentOffset = 0

def createCallback(buffer, bufferSize, sampleRate, targetSampleRate, queue):

    def callback(data, frames, time, status):
        global currentOffset
        if status:
            print("error: ", status)
        else:
            resampledData = librosa.resample(data[:, 0], sampleRate, targetSampleRate)
            # print("resampled shape: ", resampledData.shape)
            dataSize = len(resampledData)
            remainingSpace = bufferSize - currentOffset
            if remainingSpace >= dataSize:
                newOffset = currentOffset + dataSize
                buffer[currentOffset:newOffset] = resampledData[:]
                currentOffset = newOffset
            else:
                buffer[currentOffset:] = resampledData[:remainingSpace]
                queue.put(buffer.copy())
                currentOffset = dataSize - remainingSpace
                buffer[:currentOffset] = resampledData[remainingSpace:]
            # print("new Offset: ", currentOffset)
    return callback

try:
    import sounddevice as sd
    import soundfile as sf

    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        # soundfile expects an int, sounddevice provides a float:
        args.samplerate = int(device_info['default_samplerate'])
    if args.filename is None:
        args.filename = tempfile.mktemp(prefix='rec_unlimited_',
                                        suffix='.wav', dir='')
    q = queue.Queue()
    buffer = np.zeros(args.targetsamplerate)
    
    labels = helpers.readlines(args.labelsfile)
    speechToTextInferencer = SpeechToTextInferencer(args.checkpointdir, labels)

    # Make sure the file is opened before recording anything:
    with sf.SoundFile(args.filename, mode='x', samplerate=args.targetsamplerate,
                      channels=args.channels, subtype=args.subtype) as file:
        with sd.InputStream(samplerate=args.samplerate, device=args.device,
                            channels=args.channels, callback=createCallback(buffer, len(buffer), args.samplerate, args.targetsamplerate, q)):
            print('#' * 80)
            print('press Ctrl+C to stop the recording')
            print('#' * 80)
            while True:
                data = q.get()
                file.write(data)
                
                spectrogram = helpers.convertToLogMelSpectrogram(data, args.targetsamplerate)
                print("predictions: ", speechToTextInferencer.infer(spectrogram))

except KeyboardInterrupt:
    print('\nRecording finished: ' + repr(args.filename))
    parser.exit(0)
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
