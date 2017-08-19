#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 23:07:26 2017

@author: abriosi
"""

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
# this is the size of our encoded representations
encoding_dim = 6  # 6 floats 
# this is our input placeholder
input_img = Input(shape=(784,))
d = Dense(256, activation='selu')(input_img)
d = Dense(128, activation='selu')(d)
encoded = Dense(encoding_dim, activation='selu', kernel_regularizer=regularizers.l2(0.01))(d)
d = Dense(128, activation='selu')(encoded)
d = Dense(256, activation='selu')(d)
decoded = Dense(784, activation='sigmoid')(d)
# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)
# create a placeholder for an encoded (6-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
deco = autoencoder.layers[-3](encoded_input)
deco = autoencoder.layers[-2](deco)
deco = autoencoder.layers[-1](deco)
# create the decoder model
decoder = Model(encoded_input, deco)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


from keras.datasets import mnist
import numpy as np
import pandas as pd
from gmm_mml import GmmMml
from keras.callbacks import TensorBoard
(x_train, _), (x_test, _) = mnist.load_data()


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)


autoencoder.fit(x_train, x_train,
                epochs=200,
                batch_size=2056,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)

clf=GmmMml()
clf.fit(encoded_imgs)

decoded_imgs = decoder.predict(clf.sample(32))


# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

df=pd.DataFrame(encoded_imgs)
df.hist(**{'figsize':(10,10)})
plt.show()

for c in range(0,clf.bestmu.shape[0]):
    samples=[]
    for i in range(0,11):
        samples.append(np.random.multivariate_normal(clf.bestmu[c],np.swapaxes(clf.bestcov,0,2)[c]))
    samples=np.array(samples)
    decoded_imgs = decoder.predict(samples)
        
    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()