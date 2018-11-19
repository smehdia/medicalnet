# IMPORT NECESSARY PACKAGES
import numpy as np
from keras.utils import multi_gpu_model, plot_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, RepeatVector, Reshape, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from keras.utils import to_categorical
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import random
import os
import h5py

# FIXED PARAMETERS
IMG_SIZE = 256
NUM_CHANNELS = 1
NUM_GPUs = 1
NUM_CLASSES = 5

# TRAINING PARAMETERS
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 100
TOTAL_DATA_NUMBER = len(h5py.File('data/dataset_train.hdf5', 'r')['images'])


def get_model():
    model_input1 = Input(shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS))
    main_branch = Conv2D(16, (3, 3),
                  activation='relu', padding='valid')(model_input1)
    main_branch = MaxPooling2D(pool_size=(2, 2))(main_branch)
    main_branch = Conv2D(32, (3, 3), activation='relu')(main_branch)
    main_branch = MaxPooling2D(pool_size=(2, 2))(main_branch)
    main_branch = Conv2D(64, (3, 3),
                  activation='relu', padding='valid')(main_branch)

    cnn1 = Conv2D(128, (3, 3), activation='relu')(main_branch)
    cnn1 = MaxPooling2D(pool_size=(2, 2))(cnn1)
    cnn1 = Conv2D(256, (3, 3), activation='relu')(cnn1)
    cnn1 = MaxPooling2D(pool_size=(2, 2))(cnn1)
    cnn1 = Flatten()(cnn1)
    cnn1 = Dropout(0.5)(cnn1)
    cnn1 = Dense(1, activation='sigmoid', name='out1')(cnn1)
    model_input2 = RepeatVector(60 * 60)(cnn1)
    model_input2 = Reshape((60, 60, 1))(model_input2)
    model_input2 = concatenate([main_branch, model_input2], axis=3)
    cnn2 = Conv2D(16, (3, 3),
                  activation='relu', padding='valid')(model_input2)
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




class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.confusion = []
        self.precision = []
        self.recall = []
        self.f1s = []
        self.kappa = []
        self.auc = []
        # create directory for saving data
        if os.path.isdir('figures') is not True:
            os.mkdir('figures')

    def on_epoch_begin(self, epoch, logs={}):

        score = np.asarray(self.model.predict(self.validation_data[0])[1])
        integer_prediction = np.argmax(score, axis=1)
        one_hot_targets = self.validation_data[2]
        integer_targets = [np.where(r == 1)[0][0] for r in one_hot_targets]

        # calculate metrics
        self.auc.append(roc_auc_score(one_hot_targets, score))
        self.confusion.append(confusion_matrix(integer_targets, integer_prediction))
        self.precision.append(precision_score(integer_targets, integer_prediction, average='micro'))
        self.recall.append(recall_score(integer_targets, integer_prediction, average='micro'))
        self.f1s.append(f1_score(integer_targets, integer_prediction, average='micro'))
        self.kappa.append(cohen_kappa_score(integer_targets, integer_prediction))
        # save and plot metrics
        plt.plot(self.auc)
        plt.xlabel('Epochs')
        plt.title('AUC for Validation Data')
        plt.savefig('figures/auc.png')
        plt.close()
        plt.plot(self.precision)
        plt.xlabel('Epochs')
        plt.title('Precision for Validation Data')
        plt.savefig('figures/precision.png')
        plt.close()
        plt.plot(self.recall)
        plt.xlabel('Epochs')
        plt.title('Recall for Validation Data')
        plt.savefig('figures/recall.png')
        plt.close()
        plt.plot(self.kappa)
        plt.xlabel('Epochs')
        plt.title('Kappa for Validation Data')
        plt.savefig('figures/Kappa.png')
        plt.close()
        plt.plot(self.f1s)
        plt.xlabel('Epochs')
        plt.title('F1 Score for Validation Data')
        plt.savefig('figures/f1.png')
        plt.close()


        return

if __name__ == "__main__":
    # define CNNs
    model = get_model()
    # compile model and defining loss function and accuracies
    model.compile(optimizer=Adam(LEARNING_RATE),
                  loss={'out1': 'binary_crossentropy', 'out2': 'categorical_crossentropy'},
                  metrics={'out1': 'binary_accuracy', 'out2': 'categorical_accuracy'})
    # get validation data
    X_validation, y_validation_net1, y_validation_net2 = prepare_validation_data()
    # define callbacks
    # tensorboard callback for saving loss
    tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=False)
    # model checkpointer callback to save model
    checkpointer = ModelCheckpoint(filepath='model.hdf5', verbose=1, save_best_only=True)
    # Kappa score and ... Callback
    metrics = Metrics()
    # train the network
    model.fit_generator(training_data_generator(), validation_data=(X_validation, {'out1': y_validation_net1, 'out2': y_validation_net2}), steps_per_epoch=TOTAL_DATA_NUMBER//(BATCH_SIZE), nb_epoch=EPOCHS, verbose=1, nb_worker=1, callbacks=[checkpointer, tbCallBack, metrics])













