# IMPORT NECESSARY PACKAGES
import numpy as np
from keras.utils import multi_gpu_model
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, Input
from keras.models import Model

IMG_SIZE = 256
NUM_CHANNELS = 1
NUM_GPUs = 1

def get_model():
    model_input = Input(shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS))
    x = Conv2D(32, (3, 3),
                      activation='relu', padding='valid')(model_input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3),
               activation='relu', padding='valid')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)

    if NUM_GPUs <= 1:
        model = Model(inputs=[model_input], outputs=[x])
    else:
        model = multi_gpu_model(Model(inputs=[model_input], outputs=[x]), gpus=NUM_GPUs)
    return model

if __name__ == "__main__":
    model = get_model()
    model.summary()








