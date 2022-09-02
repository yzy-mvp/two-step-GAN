# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 10:24:09 2022

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
from sklearn.model_selection import KFold

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')

#plot rodar figures
def plot_data(datas,number):
    features = ['δr','∆χ','VEC','delta H','∆S','Ω','Λ','γ parameter' , 'D.χ' , 'e1/a' , 'e2/a', 'Ec' ,
                'η' , 'D.r' , 'A','F','w','G','δG','D.G','μ', 'Hardness']
    angles = np.linspace(0, 2*np.pi, len(features),endpoint=False)
    angles=np.concatenate((angles,[angles[0]]))
    features = np.concatenate((features,[features[0]]))
    
    plt.figure(figsize=(6,6),facecolor='white')
    plt.subplot(111,polar=True)
    for value in datas:
        value = np.concatenate((value,[value[0]]))
        #plt.subplot(111,polar=True)
        plt.polar(angles,value,'b-',linewidth=1,alpha=0.2)
        plt.fill(angles, value,alpha=0.25,color='g')
        plt.thetagrids(angles * 180/np.pi, features)
    
    plt.grid()
    #plt.savefig()


class GAN():
    # init 
    def __init__(self):
        self.dims = 21
        self.img_shape =(self.dims,)
        self.gen_data = None
        #Adam optimizer
        optimizer = Adam(0.0003, 0.3)

        # discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
            optimizer=optimizer,
            metrics=['accuracy'])

        #generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        z = keras.Input(shape=(10,))
        img = self.generator(z)
         
        self.discriminator.trainable = False
       
        valid = self.discriminator(img)

        # combine generator and discriminator,random vector=> generated data=> discriminate true or false 
        self.combined = keras.Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = (10,)
        
        model = keras.Sequential()

        model.add(Dense(128, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(21, activation='tanh'))
        model.add(Reshape(self.img_shape))

        #model.summary()
        noise = tf.keras.Input(shape=noise_shape)
        img = model(noise)

        return keras.Model(noise, img)

    def build_discriminator(self):

        img_shape =(self.dims,)
        model = keras.Sequential()

        model.add(Flatten(input_shape=img_shape))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        #model.summary()

        img = keras.Input(shape=img_shape)
        validity = model(img)

        return keras.Model(img, validity)

    def train(self, epochs, X_data, batch_size=64, save_interval=100):
        
        half_batch = int(batch_size / 2)
        
        d_losses, g_losses = [],[]
        for epoch in range(epochs):

            # ---------------------
            #  train discriminator
            # ---------------------

            # select half_batch size data randomly
            idx = np.random.randint(0, X_data.shape[0], half_batch)
            imgs = X_data[idx]

            noise = np.random.normal(0, 1, (half_batch, 10))

            
            gen_imgs = self.generator.predict(noise)

            # calculate discriminator loss
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # ---------------------
            #   triain generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 10))

            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)
            
            d_losses.append(d_loss[0])
            g_losses.append(g_loss)

            # display the progress log
            #print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # save data at certain intervals
            if epoch % save_interval == 0:
                noise = np.random.normal(0, 1, (batch_size, 10))
                gen_imgs = self.generator.predict(noise)
        #plot_data(gen_imgs,epoch)
        self.gen_data = gen_imgs
        
        #gen_df = pd.DataFrame(gen_imgs,columns=df.columns[:-1])
        #gen_df.to_csv(r'./Gen_fea_%d.csv'%half_batch,index=None)
                