# import necessary packages
import numpy as np
import os
from sklearn.model_selection import train_test_split
import h5py
import cv2
from keras.models import load_model

# FIXED PARAMETERS
IMG_SIZE = 256
NUM_CLASSES = 5
NUM_CHANNELS = 1
NUMBER_IMAGES_IN_EACH_CLASS = 15000
TRAINVAL_TEST_SPLIT_RATIO = 0.2
TRAIN_VAL_SPLIT_RATIO = 0.2
DATA_DIRECTORY = '../Preprocess_and_Save_Images_as_numpy_array'
H5_DATASET_DIRECTORY = 'data/dataset_train.hdf5'

def initialize_h5():
    # create h5 dataset files in data directory
    with h5py.File(H5_DATASET_DIRECTORY, 'w') as f:
        f.create_dataset('images', (1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS), maxshape=(None, IMG_SIZE, IMG_SIZE, NUM_CHANNELS), dtype='uint8')
        f.create_dataset('labels', (1, 1), maxshape=(None, 1), dtype='int32')

    return

def append_data_to_h5(dataset_dir, X, y):
    # append data to the dataset
    with h5py.File(dataset_dir, 'a') as f:
        dataset_images = f['images']
        len_dataset_before_expansion = dataset_images.len()
        dataset_images.resize((len_dataset_before_expansion+X.shape[0], IMG_SIZE, IMG_SIZE, NUM_CHANNELS))
        dataset_images[len_dataset_before_expansion:dataset_images.len()] = X
        dataset_images = f['labels']
        len_dataset_before_expansion = dataset_images.len()
        dataset_images.resize((len_dataset_before_expansion+y.shape[0], 1))
        dataset_images[len_dataset_before_expansion:dataset_images.len()] = y

    return


def prepare_dataset(directory):
    number_of_data_in_each_class = []
    print('Splitting Data to Train and Test Data')
    # initialize dataset
    initialize_h5()
    # read each numpy array related to every class
    for i in range(NUM_CLASSES):
        print('Loading Class {}...'.format(i))
        # load numpy array and reshape it
        images = np.load(directory + '/images_{}.npy'.format(i)).reshape([-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
        targets = i*np.ones(shape=(images.shape[0], 1), dtype='int32')
        # split data of the class to train and test
        X_tv, X_test, y_tv, _ = train_test_split(images, targets,
                                                            test_size=TRAINVAL_TEST_SPLIT_RATIO, random_state=42)
        # split data of the train to train and validation
        X_train, X_val, y_train, _ = train_test_split(X_tv, y_tv,
                                                            test_size=TRAIN_VAL_SPLIT_RATIO, random_state=20)

        print('Appending Data to the Dataset ... ')
        append_data_to_h5(H5_DATASET_DIRECTORY, X_train, y_train)
        np.save('data/images_test_class{}'.format(i), X_test)
        np.save('data/images_val_class{}'.format(i), X_val)
        # get number of training data in each class
        number_of_data_in_each_class.extend([X_train.shape[0]])

    return number_of_data_in_each_class

def balance_classes(number_of_data_in_each_class):
    # balancing data using GAN models
    # get class which contains maximum number of data
    maximum_class = number_of_data_in_each_class.index(max(number_of_data_in_each_class))
    for i in [x for x in range(len(number_of_data_in_each_class)) if x!=maximum_class]:
        # loading gan model
        gan_model = load_model('../DCGAN/Class{}/generator_class{}.h5'.format(i, i))
        # get number of data should be generated
        number_added_data_in_class = max(number_of_data_in_each_class) - number_of_data_in_each_class[i]
        # create random gaussian noise as the input of the gan model (Latent Dimension is 100)
        noise = np.random.normal(0, 1, (number_added_data_in_class, 100))
        print('Appending {} Fake Images from the Class {}...'.format(number_added_data_in_class, i))
        # convert generated images to 0-255
        generated_images = 255 * (gan_model.predict(noise) * 0.5 + 0.5)
        generated_images = generated_images.astype('uint8')
        # create labels for them according to their class
        targets = i * np.ones(shape=(generated_images.shape[0], 1), dtype='int32')
        # append data to the h5 dataset
        append_data_to_h5(H5_DATASET_DIRECTORY, generated_images, targets)




if __name__ == "__main__":

    # create directory for saving data
    if os.path.isdir('data') is not True:
        os.mkdir('data')

    number_of_data_in_each_class = prepare_dataset(DATA_DIRECTORY)
    print('Number of Data in Each Class:')
    print(number_of_data_in_each_class)
    balance_classes(number_of_data_in_each_class)






