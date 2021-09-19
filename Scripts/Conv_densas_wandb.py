import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import plotly.express as px
from IPython.display import clear_output

import wandb
from wandb.keras import WandbCallback

tf.random.set_seed(666)

class AutoEncoder_conv_d(tf.keras.Model):
    """
    AutoEncoder con capas convolucionales.
    """
    def __init__(self, input_shape, latent_dim):
        """
        Parameters
        ----------
        input_shape: int
            Ancho/largo de las imágenes.
        latent_dim: int
            Dimensión del espacio latente.
        """
        super(AutoEncoder_conv_d, self).__init__() #Hereda todos los métodos de tf.keras.Model
        
        #Creamos el modelo secuencial para el encoder
        self.encoder = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), #Imagen28*28
                                   input_shape=(input_shape,input_shape,1), padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), #Reducimos la imagen a la mitad 14*14
            tf.keras.layers.Conv2D(16, (3,3), padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),#Reducimos la imagen a la mitad 7*7
            # tf.keras.layers.Conv2D(8, (3,3), padding='same'),
            # tf.keras.layers.LeakyReLU(),
            # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), #Reducimos la imagen a la mitad 3*3
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(64),#Red densa
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(32),#Red densa
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(latent_dim)#Red densa de dos nodos (espacio latente)
        ])
        
        self.decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(32, input_shape=(latent_dim,)),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(64),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(128),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(7*7*16),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape((7,7,16)),
            tf.keras.layers.Conv2DTranspose(16,(3,3), padding='same', strides = (2,2)),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(32,(3,3), padding='same', strides = (2,2)),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(1,(3,3), padding='same', strides = (1,1), activation = "sigmoid"),
            # tf.keras.layers.Sigmoid()
        ])
    
    def call(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded  


def reconstruccion(model, n, data):
    """Función de reconstrucción de imágenes DESPUÉS de cada época"""

    encoded_imgs = model.encoder(data[:n]).numpy()
    decoded_imgs = model.decoder(encoded_imgs).numpy()


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
      # plt.show()


def plot_history(history):
    """Función de función del error en train y test después del entrenamiento"""

    plt.figure()
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Test")
    plt.legend()
    # plt.show()

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
            espacio_latente(model=self.model, latent_dim=self.latent_dim)



def espacio_latente(model, latent_dim):

    #Codificamos todo los datos de train y los pintamos en función de su etiqueta
    latent = model.encoder(Xtrain[:10000])
    latent_df = pd.DataFrame(latent)
    latent_df["label"] = Ytrain[:10000].astype(str)

    #Representación gráfica
    fig = px.scatter(latent_df, x = 0, y = 1, color = "label",
                     category_orders={"label": ["0","1", "2", "3", "4", "5", "6", "7", "8", "9"]})
    fig.show()

if __name__ == "__main__":

    config = dict(
        LATENT_DIM = 2,
        EPOCHS = 5,
        BATCH_SIZE = 64,
        LEARNING_RATE = 0.02
    )

    run = wandb.init(config=config, project="VAE-GAN-TEST")
    config = wandb.config

    (Xtrain, Ytrain), (Xtest, Ytest) = tf.keras.datasets.mnist.load_data()

    Xtrain = Xtrain / 255
    Xtest = Xtest / 255

    Xtrain = np.expand_dims(Xtrain,-1)
    Xtest = np.expand_dims(Xtest,-1)


    ae_conv_d = AutoEncoder_conv_d(input_shape=Xtrain.shape[1],latent_dim=config.LATENT_DIM)

    #Seleccionamos el optimizador y la función de pérdida
    ae_conv_d.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE), 
                    loss="binary_crossentropy")

    history = ae_conv_d.fit(Xtrain, Xtrain, 
                            epochs=config.EPOCHS,     
                            batch_size=config.BATCH_SIZE, 
                            shuffle=True, 
                            validation_data=(Xtest, Xtest),
                            callbacks=[CallBackReconstruccion(n=10, latent_dim=2, data=Xtrain),
                                      WandbCallback()])

    wandb.finish()

    plot_history(history)
    plt.show()