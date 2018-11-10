# IMPORT NECESSARY PACKAGES
import numpy as np
import cv2
import glob

# FIXED PARAMETERS
IMG_SIZE = 256
NUM_CLASSES = 5

if __name__ == "__main__":
        # INITIALIZE DATASET
        target_first_network = []
        target_second_network = []
        images = []
        number_of_images_in_dataset = 0
        # READ EACH IMAGE
        for classpath in glob.glob('/home/desktop/Desktop/net_medical/Classifier/train/*'):
            for imagepath in glob.glob(classpath+'/*.*'):
                # READ EACH IMAGE
                image = cv2.imread(imagepath, 0)
                # RESIZE IMAGE TO FIXED SIZE
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                # APPEND IMAGE TO THE DATASET
                images.append(image)
                # EXTRACT CLASS OF THE IMAGE  (0,1,2,3,4)
                class_image = int(classpath.split('/')[-1])
                # APPEND CLASS TO LABELS REGARDING PDR OR NPRD
                if class_image == 4:
                    target_first_network.append([0, 1])
                else:
                    target_first_network.append([1, 0])
                # CREATE ONE HOT VECTOR FOR SECOND STAGE CNN
                one_hot_class = np.zeros(shape=(NUM_CLASSES), dtype='int32')
                one_hot_class[class_image] = 1
                # ADD ONE HOT VECTOR TO THE DATASET
                target_second_network.append(one_hot_class)

                number_of_images_in_dataset += 1
                print 'Number of images in the dataset is: ' + str(number_of_images_in_dataset)

        # CONVERT LISTS TO NUMPY ARRAY
        images = np.array(images)
        target_first_network = np.array(target_first_network)
        target_second_network = np.array(target_second_network)
        # SAVE NUMPY OF DATASET
        np.save('dataset_images', images)
        np.save('targets_first_network', target_first_network)
        np.save('targets_second_network', target_second_network)
