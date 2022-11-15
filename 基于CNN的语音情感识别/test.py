import numpy as np
from keras.models import load_model
import pickle
import librosa
from data_extraction import normalizeVoiceLen, VOICE_LEN, N_FFT

model = load_model('speech_mfcc_model.h5')
paradict = {}
with open('mfcc_model_para_dict.pkl', 'rb') as f:
    paradict = pickle.load(f)
DATA_MEAN = paradict['mean']
DATA_STD = paradict['std']
emotionDict = paradict['emotion']
edr = dict([(i, t) for t, i in emotionDict.items()])


filePath = r'C:\Users\LianYu\Desktop\record1.mp3'
y, sr = librosa.load(filePath, sr=None)
y = normalizeVoiceLen(y, VOICE_LEN)  # 归一化长度
mfcc_data = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=int(N_FFT / 4))
feature = np.mean(mfcc_data, axis=0)
feature = feature.reshape((126, 1))
feature -= DATA_MEAN
feature /= DATA_STD
feature = feature.reshape((1, 126, 1))
result = model.predict(feature)
index = np.argmax(result, axis=1)[0]
print(edr[index])
