import cv2
import numpy as np
import string
from tensorflow.keras.utils import *
import math
from stn_model import create_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


char_class = string.ascii_uppercase    # A-Z
width, height, n_class = 60, 60, len(char_class)
char_list = list(char_class)


def vector(i):
    v = np.zeros(n_class)
    v[i] = 1
    return v


class SequenceData(Sequence):

    def __init__(self, data_x, data_y, batch_size):
        self.batch_size = batch_size
        self.data_x = data_x
        self.data_y = data_y
        self.indexes = np.arange(len(self.data_x))

    def __len__(self):
        return math.floor(len(self.data_x) / float(self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __getitem__(self, idx):

        batch_index = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [self.data_x[k] for k in batch_index]
        batch_y = [self.data_y[k] for k in batch_index]

        x = np.zeros((self.batch_size, height, width, 3))
        y = np.zeros((self.batch_size, n_class))

        for i in range(self.batch_size):

            img = cv2.imread(batch_x[i])
            img1 = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            img2 = img1 / 255
            x[i, :, :, :] = img2

            char = batch_y[i]
            char_index = char_class.find(char)
            y[i, :] = vector(char_index)

            # print(char, char_index)
            # print(vector(char_index))
            # cv2.namedWindow("Image")
            # cv2.imshow("Image", img2)
            # cv2.waitKey(0)

        return x, y


def train_network(train_generator, validation_generator, epoch):

    model = create_model(num_classes=n_class)

    adam = Adam(lr=1e-4, amsgrad=True)
    log_dir = "Logs/"
    checkpoint = ModelCheckpoint(log_dir + 'epoch{epoch:03d}-train_loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epoch,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint]
    )

    model.save_weights('first_weights.hdf5')


def load_network_then_train(train_generator, validation_generator, epoch, input_name, output_name):

    model = create_model()
    model.load_weights(input_name)
    print('网络层总数为：', len(model.layers))

    adam = Adam(lr=1e-4, amsgrad=True)
    log_dir = "Logs/"
    checkpoint = ModelCheckpoint(log_dir + 'epoch{epoch:03d}-train_loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epoch,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint]
    )

    model.save_weights(output_name)



