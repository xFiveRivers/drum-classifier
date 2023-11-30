from scipy.io import wavfile
import librosa

# rate, signal = wavfile.read('data/wavefiles/hi-hat_001.wav')
# sci_length = signal.shape[0] / rate
# print(sci_length)

signal, rate = librosa.load('data/wavefiles/hi-hat_001.wav', sr=None)
lib_length = signal.shape[0] / rate
print(lib_length)