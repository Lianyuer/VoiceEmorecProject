from keras import optimizers
from matplotlib import pyplot as plt

from predict import model, x_train, y_train, x_val, y_val
import keras

opt = optimizers.RMSprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor='acc',
        patience=50,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath='speechmfcc_model_checkpoint.h5',
        monitor='val_loss',
        save_best_only=True
    ),
    keras.callbacks.TensorBoard(
        log_dir='../speechmfcc_train_log'
    )
]
history = model.fit(x_train, y_train,
                    batch_size=16,
                    epochs=200,
                    validation_data=(x_val, y_val),
                    callbacks=callbacks_list)
model.save('speech_mfcc_model.h5')
model.save_weights('speech_mfcc_model_weight.h5')
# 可视化训练结果：

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
