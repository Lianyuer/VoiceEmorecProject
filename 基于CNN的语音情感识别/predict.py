import pickle

import numpy as np
from keras import layers
from keras import models
from keras.utils import to_categorical
import 基于CNN的语音情感识别.predict

# 读取特征
from 基于CNN的语音情感识别.model_SVM_wrapper import ModelSVMWrapper

mfccs = {}
with open('mfcc_feature_dict.pkl', 'rb') as f:
    mfccs = pickle.load(f)

# 设置标签
emotionDict = {}
emotionDict['angry'] = 0
emotionDict['fear'] = 1
emotionDict['happy'] = 2
emotionDict['neutral'] = 3
emotionDict['sad'] = 4
emotionDict['surprise'] = 5

data = []
labels = []
data = data + mfccs['angry']
print(len(mfccs['angry']))
for i in range(len(mfccs['angry'])):
    labels.append(0)

data = data + mfccs['fear']
print(len(mfccs['fear']))
for i in range(len(mfccs['fear'])):
    labels.append(1)

print(len(mfccs['happy']))
data = data + mfccs['happy']
for i in range(len(mfccs['happy'])):
    labels.append(2)

print(len(mfccs['neutral']))
data = data + mfccs['neutral']
for i in range(len(mfccs['neutral'])):
    labels.append(3)

print(len(mfccs['sad']))
data = data + mfccs['sad']
for i in range(len(mfccs['sad'])):
    labels.append(4)

print(len(mfccs['surprise']))
data = data + mfccs['surprise']
for i in range(len(mfccs['surprise'])):
    labels.append(5)

print(len(data))
print(len(labels))

# 设置数据维度
data = np.array(data)
data = data.reshape((data.shape[0], data.shape[1], 1))

labels = np.array(labels)
labels = to_categorical(labels)

# 数据标准化
DATA_MEAN = np.mean(data, axis=0)
DATA_STD = np.std(data, axis=0)

data -= DATA_MEAN
data /= DATA_STD
# 接下来保存好参数，模型预测的时候需要用到。

paraDict = {}
paraDict['mean'] = DATA_MEAN
paraDict['std'] = DATA_STD
paraDict['emotion'] = emotionDict
with open('mfcc_model_para_dict.pkl', 'wb') as f:
    pickle.dump(paraDict, f)
# 最后是打乱数据集并划分训练数据和测试数据。

ratioTrain = 0.8
numTrain = int(data.shape[0] * ratioTrain)
permutation = np.random.permutation(data.shape[0])
data = data[permutation, :]
labels = labels[permutation, :]

x_train = data[:numTrain]
x_val = data[numTrain:]
y_train = labels[:numTrain]
y_val = labels[numTrain:]

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)

"""定义模型

使用keras定义模型，代码如下：
"""
from keras.utils import plot_model
from keras import regularizers


# # Build a classical model
# def build_model():
model = models.Sequential()
model.add(layers.Conv1D(256, 5, activation='relu', input_shape=(126, 1)))
model.add(layers.Conv1D(128, 5, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling1D(pool_size=(8)))
model.add(layers.Conv1D(128, 5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv1D(128, 5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv1D(128, 5, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling1D(pool_size=(3)))
model.add(layers.Conv1D(256, 5, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(6, activation='softmax'))

    # # The extra metric is important for the evaluate function
    # model.compile(optimizer='rmsprop',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    # return model


# # Wrap it in the ModelSVMWrapper
# wrapper = ModelSVMWrapper(model)


plot_model(model, to_file='mfcc_model.png', show_shapes=True)
model.summary()
