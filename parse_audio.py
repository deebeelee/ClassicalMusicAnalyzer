#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 13:05:48 2019

@author: brian
"""

import os
#import subprocess
import numpy as np
import librosa as lr
import scipy as sp
import soundfile as sf
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, normalization, Conv2D,MaxPooling2D,Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras import backend as K

#
#feature_funs = {'fft': np.fft.fft,
##                'zero': lr.feature.zero_crossing_rate,
#                'spectro': lambda x : lr.stft(x,n_fft=res)}
#
#def extract_features(data,feature_list=['spectro']):
#    f = dict()
#    for label in feature_list:
#        fun = feature_funs.get(label,'notImpl')
#        if fun == 'notImpl':
#            raise ValueError('Desired feature not yet implemented or misspelled. '
#                             'Acceptable features are currently as follows:\n{0}'.format(list(feature_funs.keys())))
#        else:
#            f[label] = fun(data)
#    return f

#%% Get and preprocess data

categorical = False # If yes, use all composers, not just Bach/Debussy
intervals = [2] # seconds per slice
length = 200 # seconds to parse for; set to np.inf to make it go the whole duration
res = 256 # number of fft bins per spectrogram

all_features = dict()
for interval in intervals:
    all_features[interval] = []
if categorical:
    with open('downloaded_audio/composers.txt','r') as cfile:
        composers = [s.rstrip() for s in cfile]
else:
    composers = ['Bach','Debussy']
cdict = dict(zip(composers,range(len(composers))))
for name in composers:
    print('In group: '+name+'\n')
    for file in os.listdir('downloaded_audio/'+name+'/'):
        l = length
        filename = 'downloaded_audio/{0}/{1}'.format(name,file)
        print('loading and processing file: '+file)
        if sf.info(filename).duration <= length:
            l = sf.info(filename).duration
        num_frames = int(sf.info(filename).samplerate*l)
        waveform,fs = sf.read(filename,frames=num_frames)
        # waveform = waveform[::3]
        if waveform.ndim == 2: # stereo to mono if necessary
            waveform = sum(waveform.T)/waveform.shape[1]
        for interval in intervals:
            trim_length = len(waveform) - len(waveform) % interval*fs
            samples = np.reshape(waveform[0:trim_length],[-1,interval*fs])
            for row in samples:
#                all_features[interval].append([lr.stft(row,n_fft=res),name]) 
                all_features[interval].append([sp.signal.stft(row,nperseg=res)[2],name])
    print('')
#%%
plt.close('all')

for features in all_features.values():
    np.random.shuffle(features) # randomizes the order of the list
    n = len(features)
    n_train = round(0.9*n)
    n_test = n - n_train
    
    train = list(zip(*features[:n_train]))
    test = list(zip(*features[-n_test:]))
    
    x_train = np.array(train[0])
    x_test = np.array(test[0])
    y_train = np.array(to_categorical([cdict[i] for i in train[1]],len(composers)))
    y_test = np.array(to_categorical([cdict[i] for i in test[1]],len(composers)))
        
    earlystop = EarlyStopping(monitor='loss', min_delta=1e-4, patience=10, verbose=0, mode='auto')
    callback_list = [earlystop]
    
    # batch_size = 100 #round(len(x_train)/20)
    batch_size = 200 ### <CNN>
    epochs = 50
    # epochs = 10
    num_units = 100
    
    model = Sequential()
    # model.add(LSTM(units=num_units,input_shape=(x_train.shape[1],x_train.shape[2])))
    
    ### <CNN>
    assert(K.image_data_format()=='channels_last')
    x_train = np.abs(x_train)
    x_test = np.abs(x_test)
    # trainpeak = np.max(x_train)
    # x_train /= trainpeak
    # x_train = np.power(x_train,2)
    # x_test /= trainpeak
    # x_test = np.power(x_test,2)
    x_train = x_train.reshape(x_train.shape+(1,))
    x_test = x_test.reshape(x_test.shape+(1,))
    # model.add(Conv2D(16, (3, 3), activation="relu"))
    # model.add(Conv2D(8, (3, 3), activation="relu"))
    # model.add(MaxPooling2D(pool_size=(2, 4)))
    # model.add(Dropout(rate=0.05))
    # model.add(Conv2D(32, (3, 3), activation="relu"))
    # model.add(Conv2D(16, (3, 3), activation="relu"))
    # model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Flatten())
    ### </CNN>
    
    model.add(Dropout(rate=0.1))
    model.add(Dense(500,activation='relu'))
    model.add(normalization.BatchNormalization())
    model.add(Dense(y_train.shape[1], activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['acc'])
    
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callback_list,
                        shuffle=True,
                        validation_split=0.12)
    
    _,score = model.evaluate(x_test, y_test, batch_size=batch_size,verbose=0)
    print('Accuracy: {:4.2f}%'.format(score*100.0))    
    plt.figure()
    plt.plot([1-x for x in history.history['acc']])
    plt.plot([1-x for x in history.history['val_acc']])
    plt.title('model train vs test error, '+str(num_units)+' units')
    plt.ylabel('error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.draw()
    plt.show()

#%%
 
"""
instrumentation:
 - harpsichord: BACH
 - piano: DEBUSSY
 - clarinet: DEBUSSY
 - trombone/tuba: DEBUSSY
 - keyboard among orchestra: BACH (probably)
 - percussion (not timpani): DEBUSSY

style:
 - whole tones: DEBUSSY
 - anything beyond a 7th: DEBUSSY
 - dissonance: DEBUSSY
"""