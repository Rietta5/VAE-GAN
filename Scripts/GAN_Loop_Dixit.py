import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display
import pandas as pd
import plotly.express as px
from pathlib import Path
from matplotlib import image
from PIL import Image
import cv2
from skimage import color

import wandb
from wandb.keras import WandbCallback
tf.random.set_seed(666)



def hacedor_n(generador, n):
    """Función de reconstrucción de imágenes DESPUÉS de cada época"""

    ruido = tf.random.normal((n, 128))
    img_gen = generador(ruido).numpy()

    plt.figure(figsize=(20, 4))
    # plt.title("Generadas")
    for i in range(n):

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(img_gen[i].reshape(28,28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
      # plt.show()
    plt.tight_layout()

class CallBackHacedor(tf.keras.callbacks.Callback):

    def __init__(self, n):
        """
        Parameters
        ----------
        n: int
            Número de imágenes a reconstruir.
        data: list
            Listado original de imágenes.
        latent_dim: int
            Dimensión del espacio latente
        """
        self.n = n

    def on_epoch_end(self, epoch, logs=None):
        hacedor_n(self.model.generador, self.n)
        plt.suptitle(f"Época {epoch}")
        wandb.log({"Generador":plt})
        #plt.show()
    
    def on_train_end(self, epoch, logs=None):
        ruido = tf.random.normal((50, 128))
        img_gen = generador(ruido).numpy()
        for img in img_gen:
            plt.figure()
            plt.imshow(img.reshape((28,28)))
            plt.gray()
            wandb.log({"Muestra_GAN":plt})
            


class GAN(tf.keras.Model):
    def __init__(self, discriminador, generador):
        super(GAN, self).__init__()
        self.discriminador = discriminador
        self.generador = generador
    
    def train_step(self, X):

        ## Generar valores aleatorios

        ruido = tf.random.normal((tf.shape(X)[0], 128))

        ## Generar imágen sintética

        generadas = self.generador(ruido)

        ## Pasar las imágenes y entrenar el discriminador
        # Pasar las imágenes reales
        etiquetas_bien = tf.random.normal((tf.shape(X)[0],1), mean=1.0, stddev=0.05)

        with tf.GradientTape() as tape:
            pred_bien = self.discriminador(X)
            error_dis_bien = tf.keras.losses.BinaryCrossentropy()(etiquetas_bien, pred_bien)

        gradients = tape.gradient(error_dis_bien, self.discriminador.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.discriminador.trainable_variables))


        # Pasar las imágenes reales
        etiquetas_mal = tf.random.normal((tf.shape(X)[0],1), mean=0.0, stddev=0.05)

        with tf.GradientTape() as tape:
            pred_mal = self.discriminador(generadas)
            error_dis_mal = tf.keras.losses.BinaryCrossentropy()(etiquetas_mal, pred_mal)
        
        gradients = tape.gradient(error_dis_mal, self.discriminador.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.discriminador.trainable_variables))

        ## Entrenar el generador
        # Generar valores aleatorios
        ruido = tf.random.normal((tf.shape(X)[0], 128))

        # Entrenamiento
        etiquetas_bien = tf.ones((tf.shape(X)[0],1)) 
        with tf.GradientTape() as tape:
            # Generar imágen sintética
            generadas = self.generador(ruido)
            # Propagación
            pred_bien = self.discriminador(generadas)
            error_gen = tf.keras.losses.BinaryCrossentropy()(etiquetas_bien, pred_bien)

        gradients = tape.gradient(error_gen, self.generador.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.generador.trainable_variables))
        return {"Error Generador":error_gen, "Error Discriminador Verdaderas":error_dis_bien, "Error Discriminador Falsas":error_dis_mal}


if __name__ == "__main__":

    folder_dir = '../Data/Dixit/'
    paths = list()
    Xtrain = list()

    # iterate over files in
    # that directory
    images = Path(folder_dir).glob('*.png')
    for image in images:
        paths.append(image)

    for path in paths:
        image = Image.open(path)
        data = np.asarray(image)
        Xtrain.append(data)

    Xtrain = np.array(Xtrain)
    Xtrain = (Xtrain-175.5)/175.5

    #Cambio de tamaño
    Xtrain_resize = [cv2.resize(i, (128,256), interpolation = cv2.INTER_AREA) for i in Xtrain]
    Xtrain_resize = np.array(Xtrain_resize)
    
    #Escala de grises
    Xtrain_gray = color.rgb2gray(Xtrain_resize)

    #Especulares
    Xtrain_gray_spec1 = Xtrain_gray[:,:, ::-1, :]
    Xtrain_gray_spec2 = Xtrain_gray[:,::-1, :, :]
    Xtrain_gray_spec3 = Xtrain_gray[:,::-1, ::-1, :]

    # Conjunto final
    Xtrain = np.concatenate([Xtrain_gray,Xtrain_gray_spec1,Xtrain_gray_spec2,Xtrain_gray_spec3], axis = 0)


    for i in range(10):
        config = dict(
                LATENT_DIM = 128,
                EPOCHS = 30,
                BATCH_SIZE = 128,
                LEARNING_RATE = 0.0001
            )
        run = wandb.init(config=config, project="GAN-TEST-LOOP-NORM-DIXIT")
        config = wandb.config

        discriminador = tf.keras.models.Sequential([
            layers.Conv2D(64, (3,3), strides = 2, padding = "same" ,input_shape = Xtrain[0].shape),
            layers.LeakyReLU(alpha = 0.2),
            layers.Conv2D(128, (3,3), strides = 2, padding = "same"),
            layers.LeakyReLU(alpha = 0.2),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(1, activation  = "sigmoid")
        ])

        generador = tf.keras.models.Sequential([
            layers.Dense(8*4*128, input_shape=(128,)),
            layers.Reshape((8,4,128)),
            layers.Conv2DTranspose(128,(3,3), strides = 2, padding="same"),
            layers.LeakyReLU(alpha = 0.2),
            layers.Conv2DTranspose(128,(3,3), strides = 2, padding="same"),
            layers.LeakyReLU(alpha = 0.2),
            layers.Conv2DTranspose(128,(3,3), strides = 2, padding="same"),
            layers.LeakyReLU(alpha = 0.2),
            layers.Conv2DTranspose(256,(3,3), strides = 2, padding="same"),
            layers.LeakyReLU(alpha = 0.2),
            layers.Conv2DTranspose(256,(3,3), strides = 2, padding="same"),
            layers.LeakyReLU(alpha = 0.2),
            layers.Conv2D(1,(3,3), strides = 1, padding="same", activation = "tanh")
        ])

        

        gan = GAN(discriminador=discriminador, generador=generador)
        gan.compile(optimizer=tf.optimizers.Adam(learning_rate=config.LEARNING_RATE))
        history = gan.fit(Xtrain, batch_size=config.BATCH_SIZE, epochs=config.EPOCHS,
                        callbacks=[CallBackHacedor(n=10),WandbCallback()],
                        verbose = 0)
        wandb.finish()
