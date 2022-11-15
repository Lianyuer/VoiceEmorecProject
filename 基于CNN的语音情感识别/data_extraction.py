import librosa
import matplotlib.pyplot as plt
import numpy as np

path = r'D:\developers\_Machine_learning\casia1\liuchanhg\angry\201.wav'

y, sr = librosa.load(path, sr=None)


def normalizeVoiceLen(y, normalizedLen):
    nframes = len(y)
    y = np.reshape(y, [nframes, 1]).T
    # 归一化音频长度为2s,32000数据点
    if (nframes < normalizedLen):
        res = normalizedLen - nframes
        res_data = np.zeros([1, res], dtype=np.float32)
        y = np.reshape(y, [nframes, 1]).T
        y = np.c_[y, res_data]
    else:
        y = y[:, 0:normalizedLen]
    return y[0]


def getNearestLen(framelength, sr):
    framesize = framelength * sr
    # 找到与当前framesize最接近的2的正整数次方
    nfftdict = {}
    lists = [32, 64, 128, 256, 512, 1024]
    for i in lists:
        nfftdict[i] = abs(framesize - i)
    sortlist = sorted(nfftdict.items(), key=lambda x: x[1])  # 按与当前framesize差值升序排列
    framesize = int(sortlist[0][0])  # 取最接近当前framesize的那个2的正整数次方值为新的framesize
    return framesize


VOICE_LEN = 32000
# 获得N_FFT的长度
N_FFT = getNearestLen(0.25, sr)
# 统一声音范围为前两秒
y = normalizeVoiceLen(y, VOICE_LEN)
print(y.shape)
# 提取mfcc特征
mfcc_data = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=int(N_FFT / 4))

# 画出特征图，将MFCC可视化。转置矩阵，使得时域是水平的
plt.matshow(mfcc_data)
plt.title('MFCC')
"""上面代码的作用是加载声音，取声音的前两秒进行情感分析。getNearestLen()
函数根据声音的采样率确定一个合适的语音帧长用于傅立叶变换。然后通过librosa.feature.mfcc()
函数提取mfcc特征，并将其可视化。

下面的代码将数据集中的mfcc特征提取出来，并对每帧的mfcc取平均，将结果保存为文件。"""

# 提取特征
import os
import pickle

counter = 0
fileDirCASIA = r'D:\developers\_Machine_learning\casia1'

mfccs = {}
mfccs['angry'] = []
mfccs['fear'] = []
mfccs['happy'] = []
mfccs['neutral'] = []
mfccs['sad'] = []
mfccs['surprise'] = []
mfccs['disgust'] = []

listdir = os.listdir(fileDirCASIA)
for persondir in listdir:
    if (not r'.' in persondir):
        emotionDirName = os.path.join(fileDirCASIA, persondir)
        emotiondir = os.listdir(emotionDirName)
        for ed in emotiondir:
            if (not r'.' in ed):
                filesDirName = os.path.join(emotionDirName, ed)
                files = os.listdir(filesDirName)
                for fileName in files:
                    if (fileName[-3:] == 'wav'):
                        counter += 1
                        fn = os.path.join(filesDirName, fileName)
                        print(str(counter) + fn)
                        y, sr = librosa.load(fn, sr=None)
                        y = normalizeVoiceLen(y, VOICE_LEN)  # 归一化长度
                        mfcc_data = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=int(N_FFT / 4))
                        feature = np.mean(mfcc_data, axis=0)
                        mfccs[ed].append(feature.tolist())

with open('mfcc_feature_dict.pkl', 'wb') as f:
    pickle.dump(mfccs, f)
