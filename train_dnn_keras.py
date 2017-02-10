# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
from skimage import transform, io, color
sns.set_style("whitegrid", {'axes.grid': False})

PATH_IMG = "/home/naotoyoshida/udacity/Self-Driving-Car-Behavior-Cloning/drivedata/"
CSV_PATH = 'drivedata/driving_log.csv'
data = pd.read_csv(CSV_PATH,
                names=['img_center', 'img_left', 'img_right',
                        'steer_angle', 'throttle', 'brake', 'speed'],
                sep=', ', engine='python')

print(data[:3])
data.reindex(np.random.permutation(data.index)).reset_index(drop=True)

# remove zero-steering
msk = np.abs(data['steer_angle']) > 0.00001
data_nonzeroXc = data['img_center'][msk]
data_nonzeroXl = data['img_left'][msk]
data_nonzeroXr = data['img_right'][msk]
data_nonzeroy = data['steer_angle'][msk]

# get train-test split mask
msk = np.linspace(0,1,len(data_nonzeroy)) < 0.8

# Trainnig data with data augumentation
Xc_train_info = data_nonzeroXc[msk].as_matrix()
Xl_train_info = data_nonzeroXl.as_matrix()
Xr_train_info = data_nonzeroXr.as_matrix()
X_train_info = np.concatenate((Xc_train_info,
                               Xl_train_info,
                               Xr_train_info))
y_train_info_c = data_nonzeroy[msk].as_matrix()
y_train_info_lr = data_nonzeroy.as_matrix()
y_train_info = np.concatenate((y_train_info_c,
                               y_train_info_lr,
                               y_train_info_lr))

# Validation Data without data augumentation
X_val_info = data_nonzeroXc[~msk].as_matrix()
y_val_info = data_nonzeroy[~msk].as_matrix()

print(Xc_train_info.shape)
print(X_train_info.shape)
print(y_train_info.shape)
print(X_val_info.shape)
print(y_val_info.shape)
print("train")
print(X_train_info[:3])
print(y_train_info[:10])
print("val")
print(X_val_info[:3])
print(y_val_info[:10])

n_train = X_train_info.shape[0]
n_val = X_val_info.shape[0]

# Parameters
IMG_SIZE = (50, 100)
CHANNEL_SIZE = 3
BATCH_SIZE = 32
N_EPOCH = 1000


# Utility Functions
def get_formatted_image(img_path):
    img = color.rgb2yuv(io.imread(img_path)[50:,:,:])
    return transform.resize(image=img, output_shape=IMG_SIZE)


def get_single_data(index, info_tuple):
    data_x, data_y = info_tuple
    path = data_x[index]
    X_img = get_formatted_image(path)
    return X_img, data_y[index]

def get_whole_data(is_train):
    if is_train is True:
        n = n_train
        info_tuple = (X_train_info, y_train_info)
    else:
        n = n_val
        info_tuple = (X_val_info, y_val_info)
    X = np.ndarray(shape=(n, IMG_SIZE[0], IMG_SIZE[1], CHANNEL_SIZE)).astype(np.float32)
    Y = np.ndarray(shape=(n, 1)).astype(np.float32)
    for i in range(n):
        x, y = get_single_data(i, info_tuple)
        if CHANNEL_SIZE == 1:
            X[i,:,:,0] = x
        else:
            X[i,:,:,:] = x
        Y[i] = y
    return X, Y

X_train, y_train = get_whole_data(is_train=True)
X_val, y_val = get_whole_data(is_train=False)

# Train by Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization

model = Sequential()
model.add(Convolution2D(32, 4, 4,
                        border_mode='valid',
                        input_shape=(IMG_SIZE[0], IMG_SIZE[1], CHANNEL_SIZE),
                        dim_ordering="tf",
                        subsample=(2,2)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(64, 4, 4,
                        dim_ordering="tf",
                        subsample=(2,2)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(128, 3, 3,
                        dim_ordering="tf",
                        subsample=(1,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(400))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('tanh'))

model.summary()


from keras.optimizers import Adam
optimizer = Adam(lr=0.0001)
model.compile(loss='mean_squared_error',
              optimizer=optimizer)


from keras.preprocessing.image import ImageDataGenerator
import keras

sns.set_style("darkgrid", {'axes.grid': True})

datagen = ImageDataGenerator(
    rotation_range=30.,
    width_shift_range=0.2,
    height_shift_range=0.2,
    channel_shift_range=0.1)

# fits the model on batches with real-time data augmentation:
cbk = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
cbk_cp = keras.callbacks.ModelCheckpoint('./models/model.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
model.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True),
                    samples_per_epoch=len(X_train),
                    nb_epoch=N_EPOCH,
                    validation_data=(X_val, y_val),
                    nb_val_samples=len(X_val),
                    callbacks=[cbk, cbk_cp])

model.save('model.h5')
