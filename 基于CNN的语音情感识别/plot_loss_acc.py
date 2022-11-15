import pickle
from matplotlib import pyplot as plt

import train
from predict import model, x_train, y_train, x_val, y_val

model.compile(loss='categorical_crossentropy', optimizer=train.opt, metrics=['acc'])

history = model.fit(x_train, y_train,
                    batch_size=16,
                    epochs=200,
                    validation_data=(x_val, y_val),
                    callbacks=train.callbacks_list)

with open('log.txt', 'wb') as file_txt:
    pickle.dump(history.history, file_txt)

with open('log.txt', 'rb') as file_txt:
    history = pickle.load(file_txt)


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
