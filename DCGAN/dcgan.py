from __future__ import print_function, division
# IMPORT NECESSARY PACKAGES
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.initializers import RandomNormal
import keras.backend as K
import matplotlib.pyplot as plt
from keras.models import load_model
import sys
import numpy as np

# FIXED PARAMETERS
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 200000
SAMPLE_INTERVAL = 50
CLASS = 4
IMAGES_NUMPY_ARRAY_PATH = '../Preprocess_and_Save_Images_as_numpy_array/images_{}.npy'.format(CLASS)
# Initialize discriminator and generator loss to save them after each iteration
discriminator_loss = []
generator_loss = []

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = IMG_SIZE
        self.img_cols = IMG_SIZE
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        # Size of the latent dimension in DCGAN
        self.latent_dim = 100
        # set optimzer parameters
        optimizer_learning_rate = 0.0002
        optimizer_momentum = 0.5
        optimizer = Adam(optimizer_learning_rate, optimizer_momentum)
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        # Build the generator
        self.generator = self.build_generator()
        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    # Build generator architecture
    def build_generator(self):
        model = Sequential()
        model.add(Dense(64 * 16 * 16, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((16, 16, 64)))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(32, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(32, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(16, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(8, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=5, padding="same"))
        model.add(Activation("tanh"))
        # print model summary in the terminal
        model.summary()
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img)
    # Build discriminator architecture
    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="valid"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        img = Input(shape=self.img_shape)
        validity = model(img)
        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset for the DCGAN
        images = np.load(IMAGES_NUMPY_ARRAY_PATH)
        # Reshape them properly to feed them to the DCGAN
        images = images.reshape([-1, IMG_SIZE, IMG_SIZE, self.channels])
        X_train = images.astype('float32')
        # Rescale Inputs between -1 and 1
        X_train = (X_train - 127.5) / 127.5
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # ---------------------
            #  Train Generator
            # ---------------------
            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)
            # save generator and discriminator loss after each epoch
            discriminator_loss.extend([d_loss[0]])
            generator_loss.extend([g_loss])
            # save discriminator and generator loss after each iteration
            plt.plot(discriminator_loss)
            plt.plot(generator_loss)
            plt.savefig('Loss.png')
            plt.close()

            # Print the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        # save generator
        self.generator.save('generator_class{}.h5'.format(CLASS))
        # generate sample fake images with model
        r, c = 3, 3
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        # plot fake images
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        # save figure
        fig.savefig("images/%d.png" % epoch)
        plt.close()

if __name__ == '__main__':
    dcgan = DCGAN()
    # TRAIN THE NETWORK
    dcgan.train(epochs=EPOCHS, batch_size=BATCH_SIZE, save_interval=SAMPLE_INTERVAL)
