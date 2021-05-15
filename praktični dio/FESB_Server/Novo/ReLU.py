import tensorflow as tf
import os
import random
import PIL
import numpy as np

from tensorflow import keras
from keras.preprocessing.image import load_img, ImageDataGenerator, array_to_img, img_to_array
from keras.models import Model, load_model
from keras.layers import Input, Activation, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose, SeparableConv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint

from skimage.transform import resize
from IPython.display import Image, display
from PIL import Image, ImageOps


class FESB_MLID(keras.utils.Sequence):
    """ Helper to iterate over the data (as Numpy arrays). """

    def __init__(self, batch_size, img_width, img_height, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """ Returns tuple (input, target) correspond to batch #idx. """

        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]

        X = np.zeros((self.batch_size, img_height,
                      img_width, 3), dtype=np.float32)
        for image_index, path in enumerate(batch_input_img_paths):
            img = load_img(path)
            x_img = img_to_array(img)
            x_img = resize(x_img, (img_height, img_width, 3), mode='constant')
            X[image_index] = x_img / 255
            #X[image_index, ..., 0] = x_img / 255

        y = np.zeros((self.batch_size, img_height,
                      img_width, 3), dtype=np.float32)
        for target_index, path in enumerate(batch_target_img_paths):
            target_img = img_to_array(load_img(path))
            target_img = resize(
                target_img, (img_height, img_width, 3), mode='constant')
            y[target_index] = target_img / 255

        return X, y


def conv2d_block(input_tensor, n_filters, kernel_size=3):

    # first layer
    x = SeparableConv2D(filters=n_filters, kernel_size=(
        kernel_size, kernel_size), padding="same")(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # second layer
    x = SeparableConv2D(filters=n_filters, kernel_size=(
        kernel_size, kernel_size), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # third layer
    x = SeparableConv2D(filters=n_filters, kernel_size=(
        kernel_size, kernel_size), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def get_model(input_img, n_filters=16, dropout=0.5):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3)
    
    outputs = Conv2D(3, (1, 1), activation='softmax')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

if __name__ == "__main__":
    input_dir = "/home/veronika/radiIzvanka_QAApproved/FESB_MLID/new_data/images"
    target_dir = "/home/veronika/radiIzvanka_QAApproved/FESB_MLID/new_data/png_masks"

    img_width = 256
    img_height = 256
    batch_size = 100

    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".jpg")
        ]
    )

    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )

    val_samples = 200
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(target_img_paths)

    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]

    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]

    train_gen = FESB_MLID(batch_size, img_width, img_height,
                          train_input_img_paths, train_target_img_paths)

    val_gen = FESB_MLID(batch_size, img_width, img_height,
                        val_input_img_paths, val_target_img_paths)

    input_img = Input((img_height, img_width, 3), name='img')
    model = get_model(input_img, n_filters=16)

    model.compile(optimizer="Adam", loss="binary_crossentropy",
                  metrics=["accuracy"])

    print(model.summary())

    callbacks = [ ModelCheckpoint('relu-weights.h5', verbose=1, save_best_only=True, save_weights_only=True)]

    results = model.fit(train_gen, batch_size=batch_size, epochs=500, callbacks=callbacks, validation_data=(val_gen))
