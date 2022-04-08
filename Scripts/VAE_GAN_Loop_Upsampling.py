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

import wandb
from wandb.keras import WandbCallback
tf.random.set_seed(666)

def kl_loss(mu, logvar):
    return 0.5 * tf.reduce_sum(tf.exp(logvar) + tf.square(mu) - 1. - logvar)

def mse_loss(pred, true):
    return tf.reduce_sum(tf.square(pred-true))

class Resampling(tf.keras.layers.Layer):
    def __init__(self):
        super(Resampling, self).__init__()

    def call(self, inputs, training = None, mask = None):
        mu, log_var = inputs
        epsilon = tf.random.normal(shape = tf.shape(mu), mean = 0., stddev=1.)
        
        return mu + tf.exp(log_var / 2) * epsilon

class VAE(tf.keras.Model):
    def __init__(self, latent_dim, input_dim, beta=1):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.beta = beta

        self.encoder = tf.keras.models.Sequential([
            layers.Conv2D(256, (3,3), input_shape=(input_dim,input_dim,1), padding="same", strides = 2),
            layers.LeakyReLU(alpha = 0.2),
            layers.Conv2D(128, (3,3), padding="same", strides = 2),
            layers.LeakyReLU(alpha = 0.2),
            layers.Flatten(),
            layers.Dense(128, activation = "relu")
        ])

        self.mu = layers.Dense(self.latent_dim, name="mu")
        self.logvar = layers.Dense(self.latent_dim, name="logvar")
        self.resampling = Resampling()

        self.decoder = tf.keras.models.Sequential([
            layers.Dense(7*7*128, input_shape=(latent_dim,)),
            layers.Reshape((7,7,128)),
            layers.Conv2D(128,(3,3), strides = 1, padding="same"),
            layers.UpSampling2D(2),
            layers.LeakyReLU(alpha = 0.2),
            layers.Conv2D(256,(3,3), strides = 1, padding="same"),
            layers.UpSampling2D(2),
            layers.LeakyReLU(alpha = 0.2),
            layers.Conv2D(1,(3,3), strides = 1, padding="same", activation = "tanh")
        ])

    def encode(self, data):
        data = self.encoder(data)
        mu = self.mu(data)
        logvar = self.logvar(data)
        return self.resampling(inputs=(mu, logvar))


    def generate(self, n_samples):
        z = tf.random.normal(shape=(n_samples, self.latent_dim))
        return self.decoder(z)


    def call(self, X):
        encoded = self.encoder(X)
        mu, logvar = self.mu(encoded), self.logvar(encoded)
        z = self.resampling(inputs=(mu, logvar))
        decoded = self.decoder(z)
        return decoded

    def train_step(self, X):
        with tf.GradientTape() as tape:
            encoded = self.encoder(X)
            mu, logvar = self.mu(encoded), self.logvar(encoded)
            z = self.resampling(inputs=(mu, logvar))
            decoded = self.decoder(z)
            loss_kl = kl_loss(mu, logvar)
            # loss_mse = mse_loss(decoded, X)
            loss_mse = tf.keras.losses.MeanSquaredError()(X, decoded)

            loss_total = self.beta*loss_kl + loss_mse

        gradients = tape.gradient(loss_total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"MSE": loss_mse , "KL": loss_kl, "Total": loss_total}


# Una función para representar gráficamente imágenes sueltas

def plot_image(image):    
    plt.figure()
    plt.imshow(image.numpy().reshape((28,28)))
    #plt.show()

# La función reconstrucción que se usará en el callback

def reconstruccion(model, n, data):
    """Función de reconstrucción de imágenes DESPUÉS de cada época"""

    decoded_imgs = model(data[:n]).numpy()


    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(data[i].reshape(28,28))
        plt.title("Original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28,28))
        plt.title("Reconstruida")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    wandb.log({"Reconstruccion": plt})
        #plt.show()

# Función para representar gráficamente el espacio latente

def espacio_latente_z(model, latent_dim):

    #Codificamos todo los datos de train y los pintamos en función de su etiqueta
    latent = model.encode(Xtrain[:10000])
    
    latent_df = pd.DataFrame(latent)
    latent_df["label"] = Ytrain[:10000].astype(str)

    #Representación gráfica
    fig = px.scatter(latent_df, x = 0, y = 1, color = "label",
                     category_orders={"label": ["0","1", "2", "3", "4", "5", "6", "7", "8", "9"]})
    fig.show()
    
def espacio_latente_mu(model, latent_dim):

    #Codificamos todo los datos de train y los pintamos en función de su etiqueta
    latent = model.encoder(Xtrain[:10000])
    latent_mu = model.mu(latent)
    
    latent_df = pd.DataFrame(latent_mu)
    latent_df["label"] = Ytrain[:10000].astype(str)

    #Representación gráfica
    fig = px.scatter(latent_df, x = 0, y = 1, color = "label",
                     category_orders={"label": ["0","1", "2", "3", "4", "5", "6", "7", "8", "9"]})
    fig.show()          


# Definimos el Callback para que muestre las imágenes reconstruidas junto con las imágenes 
# iniciales para apreciar los cambios.

class CallBackReconstruccion(tf.keras.callbacks.Callback):

    def __init__(self, n, data, latent_dim):
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
        self.data = data
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        reconstruccion(self.model, self.n, self.data)
        plt.suptitle(f"Época {epoch}")

    def on_train_end(self, logs=None):
        if self.latent_dim == 2:
            espacio_latente_z(model=self.model, latent_dim=self.latent_dim)
            espacio_latente_mu(model=self.model, latent_dim=self.latent_dim)

# La función que representará la evolucion de la función de coste en train y test

def plot_history(history):
    """Función de función del error en train y test después del entrenamiento"""

    plt.figure()
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Test")
    plt.legend()
    #plt.show()

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
        #plt.show()
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
            wandb.log({"Muestra_VAEGAN":plt})

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

    (Xtrain, Ytrain), (Xtest, Ytest) = tf.keras.datasets.mnist.load_data()
    Xtrain.shape, Ytrain.shape, Xtest.shape, Ytest.shape

    Xtrain = np.expand_dims(Xtrain,-1)
    Xtest = np.expand_dims(Xtest,-1)
    Xtrain.shape, Xtest.shape

    Xtrain = Xtrain / 255
    Xtest = Xtest / 255

    for i in range(20):
        config = dict(
                LATENT_DIM = 128,
                EPOCHS_VAE = 10,
                EPOCHS_GAN = 30,
                BATCH_SIZE = 128,
                LEARNING_RATE = 0.00005,
                BETA = 0.000001
            )
        
        run = wandb.init(config=config, project="VAE-GAN-TEST-LOOP")
        config = wandb.config
       
        vae = VAE(latent_dim=config.LATENT_DIM, input_dim=Xtrain.shape[1], beta=config.BETA)

        vae.compile(optimizer="adam")

        history = vae.fit(Xtrain, batch_size=config.BATCH_SIZE, epochs=config.EPOCHS_VAE, verbose=0,
                    callbacks=[CallBackReconstruccion(n=10, latent_dim=128, data=Xtrain), WandbCallback()])

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
            layers.Dense(7*7*128, input_shape = (128,)),
            layers.Reshape((7,7,128)),
            layers.Conv2D(128, (3,3), strides = 1, padding = "same"),
            layers.UpSampling2D(2),
            layers.LeakyReLU(alpha = 0.2),
            layers.Conv2D(256, (3,3), strides = 1, padding = "same"),
            layers.UpSampling2D(2),
            layers.LeakyReLU(alpha = 0.2),
            layers.Conv2D(1, (3,3), strides = 1, padding = "same", activation = "tanh")
        ])

        # Cargando los pesos preentrenados
        for layer_generador, layer_vae in zip(generador.layers, vae.decoder.layers):
            layer_generador.set_weights(layer_vae.get_weights())

        # Entrenamos la GAN
        gan = GAN(discriminador=discriminador, generador=generador)
        gan.compile(optimizer=tf.optimizers.Adam(learning_rate=config.LEARNING_RATE))
        history = gan.fit(Xtrain, batch_size=config.BATCH_SIZE, epochs=config.EPOCHS_GAN, verbose=0,
                    callbacks=[CallBackHacedor(n=10), WandbCallback()])

        wandb.finish()

