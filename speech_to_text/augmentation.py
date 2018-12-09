import librosa
import numpy as np

def augmentPitch(data, sampleRate):
	y_pitch = data.copy()
	bins_per_octave = 24
	pitch_pm = 4
	pitch_change = pitch_pm * 2 * (np.random.uniform() - 0.5)
	print("pitch_change = ",pitch_change)
	return librosa.effects.pitch_shift(y_pitch.astype('float64'),
        	sampleRate, n_steps=pitch_change,
               	bins_per_octave=bins_per_octave)

def augmentSpeed(data, sampleRate):
	y_speed = data.copy()
	speed_change = np.random.uniform(low=0.9,high=1.1)
	print("speed_change = ", speed_change)
	tmp = librosa.effects.time_stretch(y_speed.astype('float64'), speed_change)
	minlen = min(y_speed.shape[0], tmp.shape[0])
	y_speed *= 0
	y_speed[0:minlen] = tmp[0:minlen]
	return y_speed

def augmentPitchAndSpeed(data, sampleRate):
	y_pitch_speed = data.copy()
	# you can change low and high here
	length_change = np.random.uniform(low=0.5,high=1.5)
	speed_fac = 1.0  / length_change
	print("resample length_change = ", length_change)
	tmp = np.interp(np.arange(0,len(y_pitch_speed),speed_fac),np.arange(0,len(y_pitch_speed)),y_pitch_speed)
	minlen = min(y_pitch_speed.shape[0], tmp.shape[0])
	y_pitch_speed *= 0
	y_pitch_speed[0:minlen] = tmp[0:minlen]
	return y_pitch_speed

def augmentDistributionNoise(data, sampleRate):
	y_noise = data.copy()
	# you can take any distribution from https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html
	noise_amp = 0.005 * np.random.uniform() * np.amax(y_noise)
	return  y_noise.astype('float64') + noise_amp * np.random.normal(size=y_noise.shape[0])

def augmentRandomShift(data, sampleRate):
	y_shift = data.copy()
	timeshift_fac = 0.2 * 2 * (np.random.uniform()-0.5)  # up to 20% of length
	print("timeshift_fac = ", timeshift_fac)
	start = int(y_shift.shape[0] * timeshift_fac)
	if (start > 0):
    		y_shift = np.pad(y_shift,(start,0),mode='constant')[0:y_shift.shape[0]]
	else:
		y_shift = np.pad(y_shift,(0,-start),mode='constant')[0:y_shift.shape[0]]
	return y_shift

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

def applyAugmentation(data, sampleRate):
	augmentedPitchAndSpeedData = augmentPitchAndSpeed(data, sampleRate)
	augmentedUpTo1SecData = augmentFillUpTo(augmentedPitchAndSpeedData, sampleRate, 1)
	augmentedWithDistributionNoiseData = augmentDistributionNoise(augmentedUpTo1SecData, sampleRate)
	augmentedRandomShift = augmentRandomShift(augmentedWithDistributionNoiseData, sampleRate)
	return augmentedRandomShift
