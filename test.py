import librosa
import librosa.display
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

plt.figure(1,figsize=(15, 4))
grid = plt.GridSpec(1, 16)
plt.subplot(grid[0,:5])
y, sr = librosa.load('Denn alles.wav')
S=librosa.feature.melspectrogram(y=y, sr=sr)
librosa.display.specshow(librosa.power_to_db(S,ref=np.max),
                          y_axis='mel', fmax=8000,
                          x_axis='time')
plt.subplot(grid[0,5:10])
y, sr = librosa.load('Bach/Bach_Brandenburg_5_1_027.opus')
# y, sr = librosa.load('downloaded_audio/Bach/Bach_Brandenburg_5_1_027.opus')
S=librosa.feature.melspectrogram(y=y, sr=sr)
librosa.display.specshow(librosa.power_to_db(S,ref=np.max),
                          # y_axis='mel',
                          fmax=8000,
                          x_axis='time')

plt.subplot(grid[0,10:])
y, sr = librosa.load('Debussy/Debussy_Arabesque_1_2_003.opus')
# y, sr = librosa.load('downloaded_audio/Bach/Bach_Brandenburg_5_1_027.opus')
S=librosa.feature.melspectrogram(y=y, sr=sr)
librosa.display.specshow(librosa.power_to_db(S,ref=np.max),
                          # y_axis='mel',
                          fmax=8000,
                          x_axis='time')
plt.colorbar(format='%+2.0f dB')
# plt.suptitle('Mel spectrograms of Speech and audio')
# plt.tight_layout()
plt.show()