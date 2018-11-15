# IMPORT NECESSARY PACKAGES
import numpy as np
from keras.utils import multi_gpu_model, plot_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, RepeatVector, Reshape, concatenate
from keras.models import Model
from keras.optimizers import Adam
import cv2
from keras.utils import to_categorical, HDF5Matrix
import random
import h5py
# FIXED PARAMETERS
IMG_SIZE = 256
NUM_CHANNELS = 1
NUM_GPUs = 1
NUM_CLASSES = 5
TOTAL_DATA_NUMBER = len(h5py.File('data/dataset_train.hdf5', 'r')['images'])

# TRAINING PARAMETERS
LEARNING_RATE = 1e-3
BATCH_SIZE = 64


def get_model():
    model_input1 = Input(shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS))
    cnn1 = Conv2D(16, (3, 3),
                  activation='relu', padding='valid')(model_input1)
    cnn1 = MaxPooling2D(pool_size=(2, 2))(cnn1)
    cnn1 = Conv2D(32, (3, 3), activation='relu')(cnn1)
    cnn1 = MaxPooling2D(pool_size=(2, 2))(cnn1)
    cnn1 = Conv2D(64, (3, 3),
                  activation='relu', padding='valid')(cnn1)
    cnn1 = MaxPooling2D(pool_size=(2, 2))(cnn1)
    cnn1 = Conv2D(128, (3, 3), activation='relu')(cnn1)
    cnn1 = MaxPooling2D(pool_size=(2, 2))(cnn1)
    cnn1 = Conv2D(256, (3, 3), activation='relu')(cnn1)
    cnn1 = MaxPooling2D(pool_size=(2, 2))(cnn1)
    cnn1 = Flatten()(cnn1)
    cnn1 = Dropout(0.5)(cnn1)
    cnn1 = Dense(1, activation='sigmoid', name='out1')(cnn1)
    model_input2 = RepeatVector(IMG_SIZE * IMG_SIZE)(cnn1)
    model_input2 = Reshape((IMG_SIZE, IMG_SIZE, 1))(model_input2)
    model_input2 = concatenate([model_input1, model_input2], axis=3)
    cnn2 = Conv2D(16, (3, 3),
                  activation='relu', padding='valid')(model_input2)
    cnn2 = MaxPooling2D(pool_size=(2, 2))(cnn2)
    cnn2 = Conv2D(32, (3, 3), activation='relu')(cnn2)
    cnn2 = MaxPooling2D(pool_size=(2, 2))(cnn2)
    cnn2 = Conv2D(64, (3, 3),
                  activation='relu', padding='valid')(cnn2)
    cnn2 = MaxPooling2D(pool_size=(2, 2))(cnn2)
    cnn2 = Conv2D(128, (3, 3), activation='relu')(cnn2)
    cnn2 = MaxPooling2D(pool_size=(2, 2))(cnn2)
    cnn2 = Conv2D(256, (3, 3), activation='relu')(cnn2)
    cnn2 = MaxPooling2D(pool_size=(2, 2))(cnn2)
    cnn2 = Flatten()(cnn2)
    cnn2 = Dropout(0.5)(cnn2)
    cnn2 = Dense(32, activation='relu')(cnn2)
    cnn2 = Dense(5, activation='softmax', name='out2')(cnn2)

    if NUM_GPUs <= 1:
        model = Model(inputs=[model_input1], outputs=[cnn1, cnn2])
    else:
        model = multi_gpu_model(Model(inputs=[model_input1], outputs=[cnn1, cnn2]), gpus=NUM_GPUs)

    plot_model(model, to_file='model.png', show_layer_names=True, show_shapes=True)
    model.summary()

    return model

def prepare_validation_data():
    # initializing validation data
    X_validation = []
    y_validation_net1 = []
    y_validation_net2 = []
    # loop on the classes
    for i in range(NUM_CLASSES):
        # load numpy of the validation dataset
        validation_images = np.load('data/images_val_class{}.npy'.format(i)).reshape([-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
        X_validation.extend(validation_images)
        # create labels
        labels = i * np.ones(shape=(validation_images.shape[0], 1), dtype='float32')
        # convert labels to one hot matrix for CNN 2
        y_validation_net2.extend(to_categorical(labels, num_classes=5))
        # create labels for the first CNN (Binary Classifier)
        if i == 4:
            y_validation_net1.extend(np.ones(shape=(validation_images.shape[0], 1), dtype='int32'))
        else:
            y_validation_net1.extend(np.zeros(shape=(validation_images.shape[0], 1), dtype='int32'))

    # Scale Data
    X_validation = np.array(X_validation).astype('float32').reshape([-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
    X_validation /= 255.
    y_validation_net1 = np.array(y_validation_net1).astype('float32').reshape([-1, 1])
    y_validation_net2 = np.array(y_validation_net2).astype('float32').reshape([-1, NUM_CLASSES])

    return X_validation, y_validation_net1, y_validation_net2

def training_data_generator():
    while 1:
        for i in range(TOTAL_DATA_NUMBER//BATCH_SIZE):
            # generate random batch of data from dataset
            random_indices = random.sample(range(TOTAL_DATA_NUMBER-1), BATCH_SIZE)
            random_indices.sort()
            with h5py.File('data/dataset_train.hdf5', 'r') as f:
                images = f['images'][random_indices]
                labels = f['labels'][random_indices]

            images = images.astype('float32').reshape([-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
            targets_net1 = np.copy(labels)
            targets_net1[targets_net1 != 4] = 0
            targets_net1[targets_net1 == 4] = 1
            targets_net2 = to_categorical(labels, num_classes=5)

            yield images, {'out1':targets_net1, 'out2':targets_net2}


if __name__ == "__main__":
    # define CNNs
    model = get_model()
    # compile model and defining loss function and accuracies
    model.compile(optimizer=Adam(LEARNING_RATE),
                  loss={'out1': 'binary_crossentropy', 'out2': 'categorical_crossentropy'},
                  metrics={'out1': 'binary_accuracy', 'out2': 'categorical_accuracy'})
    # get validation data
    X_validation, y_validation_net1, y_validation_net2 = prepare_validation_data()
    # train the network
    model.fit_generator(training_data_generator(), validation_data=(X_validation, {'out1': y_validation_net1, 'out2': y_validation_net2}), steps_per_epoch=TOTAL_DATA_NUMBER//BATCH_SIZE, nb_epoch=2, verbose=1, nb_worker=1)












