# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 14:15:18 2022

@author: swaggy.p
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras.layers import Dense,BatchNormalization,Flatten,LeakyReLU,Reshape
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import KFold, train_test_split



class GAN():
 
    def __init__(self):
        self.dims = 11
        self.img_shape =(self.dims,)
        self.gen_data = None
   
        optimizer = Adam(0.001, 0.3)


        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
            optimizer=optimizer,
            metrics=['accuracy'])

        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        z = keras.Input(shape=(10,))
        img = self.generator(z)
         
        self.discriminator.trainable = False
       
        valid = self.discriminator(img)

        self.combined = keras.Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        noise_shape = (10,)        
        model = keras.Sequential()

        model.add(Dense(55, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.24))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(35))
        model.add(LeakyReLU(alpha=0.24))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(25))
        model.add(LeakyReLU(alpha=0.24))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(11, activation='tanh'))
        model.add(Reshape(self.img_shape))

        noise = tf.keras.Input(shape=noise_shape)
        img = model(noise)

        return keras.Model(noise, img)

    def build_discriminator(self):
        img_shape =(self.dims,)
        model = keras.Sequential()

        model.add(Flatten(input_shape=img_shape))
        model.add(Dense(60))
        model.add(LeakyReLU(alpha=0.25))
        model.add(Dense(48))
        model.add(LeakyReLU(alpha=0.25))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.25))
        model.add(Dense(1, activation='sigmoid'))

        img = keras.Input(shape=img_shape)
        validity = model(img)

        return keras.Model(img, validity)

    def train(self, epochs, X_data, batch_size=64, save_interval=100):
        
        half_batch = int(batch_size / 2)
        
        d_losses, g_losses = [],[]
        for epoch in range(epochs):
            # ---------------------
            # 
            # ---------------------
            #
            idx = np.random.randint(0, X_data.shape[0], half_batch)
            imgs = X_data[idx]

            noise = np.random.normal(0, 1, (half_batch, 10))        
            gen_imgs = self.generator.predict(noise)

        
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)           
            # ---------------------
            # ---------------------
            noise = np.random.normal(0, 1, (batch_size, 10))

            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)
            
            d_losses.append(d_loss[0])
            g_losses.append(g_loss)

            # Plot the progress
            #print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

           
            if epoch % save_interval == 0:
                noise = np.random.normal(0, 1, (batch_size, 10))
                gen_imgs = self.generator.predict(noise)
        self.gen_data = gen_imgs
        