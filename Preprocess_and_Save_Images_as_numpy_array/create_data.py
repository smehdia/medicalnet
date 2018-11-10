# IMPORT NECESSARY PACKAGES
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import re
import random

# FIXED PARAMETERS
#####
CSV_PATH = '/media/desktop/SP PHD U3/Diabetic_Kaggle/trainLabels.csv'
IMAGES_PATH = '/media/desktop/SP PHD U3/Diabetic_Kaggle/train'
#####


# apply gamma correction on the images in order to enhance contrast
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
# resize image to fixed size images
def resize(image, target_size):

    h, w = image.shape[0], image.shape[1]
    fx = target_size/float(w)
    fy = target_size/float(h)
    new_image = cv2.resize(image, None, fx=min(fx, fy), fy=min(fx, fy))
    top = np.abs(new_image.shape[0]-target_size)
    new_image = cv2.copyMakeBorder(new_image, top=top, bottom=0, left=np.abs(new_image.shape[1]-target_size), right=0,
                                borderType=cv2.BORDER_CONSTANT, value=(127, 127, 127))
    return new_image
# crop retina from the image
def crop_image(image):
    # binarize image
    _, thresholded_image = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 255,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # find contours in the image
    _, contours, _ = cv2.findContours(np.copy(thresholded_image), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # sort contorus in order to keep just retina
    contours.sort(key=cv2.contourArea, reverse=True)
    biggest_contour = contours[0]
    x, y, w, h = cv2.boundingRect(biggest_contour)
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image
# scale image regarding to its retina size
def scaleRadius(img, scale):
    x = img[img.shape[0]/2,:,:].sum(1)
    r = (x>x.mean()/10).sum()/2
    s = scale * 1.0 / r
    return cv2.resize(img, (0,0), fx=s, fy=s)
def preprocess_image(image, scale):
    # scale image regarding to its retina size
    image = scaleRadius(image, scale)
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), scale / 30), -4, 128)
    b = np.zeros(image.shape)
    cv2.circle(b, (image.shape[1] / 2, image.shape[0] / 2), int(scale * 0.9), (1, 1, 1), -1, 8, 0)
    image = image * b + 128 * (1 - b)
    image = image.astype('uint8')
    return image

if __name__ == "__main__":
    # initialzie images list
    images = []
    labels = []

    # read csv file in order to read images and save them
    f = open(CSV_PATH)
    f.readline()
    contents = f.readlines()
    # get total number of images
    total_contents = len(contents)
    # get images list
    list_images = glob.glob(IMAGES_PATH + '/*')
    for i, line in enumerate(contents):
        # read image
        image_name = line.split(',')[0]
        label = line.split(',')[1].strip()
        filename = filter(lambda x:re.search(r'.*/{}\.*'.format(image_name), x), list_images)[0]
        image = cv2.imread(filename, 1)
        # preprocess image
        image = crop_image(image)
        image = preprocess_image(image, 256)
        image = resize(image, 256)
        # add image to the list
        images.append(image)
        labels.append(int(label))
        print "Number of Saved Images: {} / {} ".format(i, total_contents)

    # convert lists to numpy array
    images = np.array(images)
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    # save each class numpy array
    for i in range(unique_labels.shape[0]):
        images_class_i = images[labels == unique_labels[i]]
        np.save('images_{}'.format(unique_labels[i]), images_class_i)
        print 'Number of Saved Images in Class {} is {}'.format(unique_labels[i], images_class_i.shape[0])
